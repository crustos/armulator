"""
AArch64 stage 1 address translation for the EL1&0 regime.

A virtual address is split into a page offset and a series of table indices. Each index
selects a descriptor; a table descriptor points at the next level down, a block or page
descriptor ends the walk and supplies the output address together with the permissions
and memory attributes for the region.

Three things make this fiddly enough to be worth spelling out:

* **Which base register.** The top bits of the VA choose between TTBR0 and TTBR1 - all
  zeros picks the low half, all ones the high half. Anything in between is a fault, which
  is what creates the enormous unmapped gap in the middle of the 64-bit address space.

* **Where the walk starts.** The starting level falls out of the configured input size
  and the granule, so a smaller address space simply skips the upper levels rather than
  walking through near-empty tables.

* **Permissions accumulate.** Table descriptors carry their own restrictions, and those
  combine with the ones on the final descriptor. A table marked read-only makes
  everything below it read-only regardless of what the leaf says.

Stage 2 translation is not implemented: it belongs to EL2, and the EL1&0 regime this
models never uses it.
"""

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.armv6.memory_attributes import MemoryAttributes, MemType
from armulator.armv8.bits_ops import bit_at, lower_chunk, substring
from armulator.armv8.enums import EL

#: Fault kinds, as encoded in the top bits of ESR_ELx.ISS.DFSC.
FAULT_ADDRESS_SIZE = 0b00
FAULT_TRANSLATION = 0b01
FAULT_ACCESS_FLAG = 0b10
FAULT_PERMISSION = 0b11

#: TCR_EL1.TG0 encoding -> granule size in bits.
TG0_GRANULE = {0b00: 12, 0b01: 16, 0b10: 14}
#: TCR_EL1.TG1 uses a different encoding for the same three sizes.
TG1_GRANULE = {0b01: 14, 0b10: 12, 0b11: 16}

#: Granule size in bits -> the highest level at which a block descriptor is allowed.
#: With a 4KB granule blocks exist at levels 1 and 2; the larger granules only at 2.
FIRST_BLOCK_LEVEL = {12: 1, 14: 2, 16: 2}

#: TCR_EL1.IPS -> supported physical address size in bits.
IPS_SIZES = {0b000: 32, 0b001: 36, 0b010: 40, 0b011: 42, 0b100: 44, 0b101: 48, 0b110: 52}


class TranslationFault(Exception):
    """
    Raised inside the walk and converted by the caller into a data or instruction abort.

    Carrying the kind and level separately means the ESR syndrome can be built once, at
    the point where we know whether the access was a fetch or a load or a store.
    """

    def __init__(self, kind, level, address, message='', stage2=False):
        super().__init__(message or f'translation fault kind={kind} level={level}')
        self.kind = kind
        self.level = level
        self.address = address
        #: True when the fault came from the hypervisor's tables rather than the
        #: guest's, which decides both where it is reported and who has to fix it.
        self.stage2 = stage2
        #: The intermediate address, for HPFAR_EL2.
        self.intermediate_address = None

    @property
    def status(self) -> int:
        """The DFSC/IFSC value for ESR_ELx.ISS."""
        return (self.kind << 2) | (self.level & 0b11)


class TlbEntry:
    """
    One cached translation, held per 4KB of virtual address regardless of the block size
    that produced it. Permissions are stored rather than a yes/no answer, so a read and a
    write to the same page can share an entry.
    """

    __slots__ = ('physical_address', 'ap', 'uxn', 'pxn', 'attr_index', 'nG',
                 'shareability', 'ns', 'level')

    def __init__(self, physical_address, ap, uxn, pxn, attr_index, nG, shareability, ns, level):
        self.physical_address = physical_address
        self.ap = ap
        self.uxn = uxn
        self.pxn = pxn
        self.attr_index = attr_index
        self.nG = nG
        self.shareability = shareability
        self.ns = ns
        self.level = level


class Mmu:
    """
    Stage 1 translation for the EL1&0 regime.

    The TLB here exists for speed rather than fidelity: a table walk costs several memory
    reads, and doing that in Python for every access makes even small firmware crawl. It
    is flushed wholesale whenever anything that could change a translation is written,
    which is conservative but never wrong.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tlb = {}
        #: Counts, useful when checking that the cache is actually doing something.
        self.walks = 0
        self.hits = 0

    def flush(self):
        """Invalidate the whole TLB. Called on TLBI and on writes to translation control."""
        self.tlb.clear()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def tcr(self) -> int:
        return self.processor.registers.regime_register('tcr')

    def ttbr(self, high: bool) -> int:
        return self.processor.registers.regime_register('ttbr1' if high else 'ttbr0')

    @property
    def mair(self) -> int:
        return self.processor.registers.regime_register('mair')

    @property
    def vtcr(self) -> int:
        return self.processor.registers.get_system_register(0b11, 0b100, 0b0010, 0b0001, 0b010)

    @property
    def vttbr(self) -> int:
        return self.processor.registers.get_system_register(0b11, 0b100, 0b0010, 0b0001, 0b000)

    def _regime(self, address):
        """
        Choose between TTBR0 and TTBR1 from the top of the address, and return the
        configuration for that half.

        Returns (high, tsz, granule_bits, base, disabled).
        """
        tcr = self.tcr
        top_bit = bit_at(address, 55)
        if self.processor.registers.regime() is not EL.EL1:
            # Only the EL1&0 regime has two halves. EL2 and EL3 translate a single
            # region through TTBR0, so a high address there is simply out of range.
            top_bit = 0

        if top_bit:
            tsz = substring(tcr, 21, 16)
            granule = TG1_GRANULE.get(substring(tcr, 31, 30))
            disabled = bool(bit_at(tcr, 23))       # EPD1
        else:
            tsz = substring(tcr, 5, 0)
            granule = TG0_GRANULE.get(substring(tcr, 15, 14))
            disabled = bool(bit_at(tcr, 7))        # EPD0

        return top_bit, tsz, granule, self.ttbr(top_bit), disabled

    def _top_byte_ignored(self, high: bool) -> bool:
        return bool(bit_at(self.tcr, 38 if high else 37))

    # ------------------------------------------------------------------
    # The walk
    # ------------------------------------------------------------------

    def _read_descriptor(self, address):
        """
        Read one 64-bit descriptor. The walk reads physical memory directly: translating
        the descriptor address would be circular.
        """
        descriptor_address = AddressDescriptor()
        descriptor_address.paddress.physicaladdress = address
        descriptor_address.paddress.ns = 1
        return self.processor.mem[descriptor_address, 8]

    def walk(self, address):
        """
        Walk the tables for ``address`` and return a :class:`TlbEntry`.

        Raises :class:`TranslationFault` for an invalid descriptor, an address outside
        the configured region, or a clear access flag.
        """
        self.walks += 1
        high, tsz, granule, base, disabled = self._regime(address)

        if granule is None:
            raise TranslationFault(FAULT_TRANSLATION, 0, address, 'reserved granule size')
        if disabled:
            raise TranslationFault(FAULT_TRANSLATION, 0, address, 'translation disabled for this region')
        if not 16 <= tsz <= 39 + 8:
            raise TranslationFault(FAULT_TRANSLATION, 0, address, 'TxSZ out of range')

        input_size = 64 - tsz

        # Everything above the configured input size must match the half we selected:
        # all zeros for TTBR0, all ones for TTBR1. This is what makes the middle of the
        # address space unmapped rather than aliasing one of the two halves.
        top = substring(address, 63, input_size) if input_size < 64 else 0
        expected = ((1 << (64 - input_size)) - 1) if high else 0
        if input_size < 64 and top != expected:
            raise TranslationFault(FAULT_TRANSLATION, 0, address, 'address outside the translated region')

        stride = granule - 3
        # The first level is whichever one still has bits left to resolve. A smaller
        # address space therefore starts further down rather than walking near-empty
        # upper tables.
        first_level = 4 - (1 + (input_size - granule - 1) // stride)
        level = first_level
        first_block_level = FIRST_BLOCK_LEVEL[granule]

        # The first-level table is only as large as the bits it has to resolve, so its
        # base is aligned to that size rather than to a full granule.
        first_index_bits = input_size - (granule + stride * (3 - first_level))
        base_address = base & ((1 << 48) - 1)
        base_address &= ~((1 << max(first_index_bits + 3, 3)) - 1)

        # Permissions picked up from table descriptors on the way down.
        table_ap = 0
        table_uxn = False
        table_pxn = False
        table_ns = False

        while True:
            low_bit = granule + stride * (3 - level)
            index_bits = first_index_bits if level == first_level else stride
            index = substring(address, low_bit + index_bits - 1, low_bit)

            descriptor = self._read_descriptor(base_address + (index << 3))

            if not bit_at(descriptor, 0):
                raise TranslationFault(FAULT_TRANSLATION, level, address, 'invalid descriptor')

            is_table = bit_at(descriptor, 1) and level < 3
            is_page = bit_at(descriptor, 1) and level == 3
            is_block = not bit_at(descriptor, 1) and level < 3

            if is_block and level < first_block_level:
                raise TranslationFault(FAULT_TRANSLATION, level, address, 'block not permitted at this level')
            if not bit_at(descriptor, 1) and level == 3:
                # At the last level bits[1:0] of 01 is reserved, not a block.
                raise TranslationFault(FAULT_TRANSLATION, level, address, 'reserved descriptor at level 3')

            if is_table:
                # Table descriptors restrict everything beneath them; the restrictions
                # accumulate as the walk descends and can never be relaxed lower down.
                table_ap |= substring(descriptor, 62, 61)
                table_uxn = table_uxn or bool(bit_at(descriptor, 60))
                table_pxn = table_pxn or bool(bit_at(descriptor, 59))
                table_ns = table_ns or bool(bit_at(descriptor, 63))
                base_address = substring(descriptor, 47, granule) << granule
                level += 1
                continue

            # A block or page ends the walk.
            if not bit_at(descriptor, 10):
                # The access flag is managed by software on this core, so a clear flag
                # is a fault rather than something the hardware quietly sets.
                raise TranslationFault(FAULT_ACCESS_FLAG, level, address, 'access flag clear')

            output_base = substring(descriptor, 47, low_bit) << low_bit
            offset = lower_chunk(address, low_bit)
            physical_address = output_base | offset

            ap = substring(descriptor, 7, 6)
            uxn = bool(bit_at(descriptor, 54)) or table_uxn
            pxn = bool(bit_at(descriptor, 53)) or table_pxn
            ns = bool(bit_at(descriptor, 5)) or table_ns

            # APTable can only remove access: bit 1 removes EL0 access, bit 0 makes the
            # region read-only.
            if bit_at(table_ap, 1):
                ap &= ~0b01
            if bit_at(table_ap, 0):
                ap |= 0b10

            return TlbEntry(
                physical_address=physical_address,
                ap=ap,
                uxn=uxn,
                pxn=pxn,
                attr_index=substring(descriptor, 4, 2),
                nG=bool(bit_at(descriptor, 11)),
                shareability=substring(descriptor, 9, 8),
                ns=ns,
                level=level,
            )

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def _check_permission(self, entry, address, is_write, is_instruction, el):
        """
        Apply AP, the execute-never bits and SCTLR_EL1.WXN.
        """
        read_only = bit_at(entry.ap, 1)
        el0_access = bit_at(entry.ap, 0)

        if is_instruction:
            execute_never = entry.uxn if el == EL.EL0 else entry.pxn
            if el == EL.EL0 and not el0_access:
                raise TranslationFault(FAULT_PERMISSION, entry.level, address,
                                       'no EL0 access to this region')
            if execute_never:
                raise TranslationFault(FAULT_PERMISSION, entry.level, address,
                                       'execute never')
            # SCTLR_EL1.WXN makes any writable region non-executable.
            if bit_at(self.processor.registers.sctlr_el1, 19) and not read_only:
                raise TranslationFault(FAULT_PERMISSION, entry.level, address,
                                       'writable region is not executable with WXN set')
            return

        if el == EL.EL0 and not el0_access:
            raise TranslationFault(FAULT_PERMISSION, entry.level, address,
                                   'no EL0 access to this region')
        if is_write and read_only:
            raise TranslationFault(FAULT_PERMISSION, entry.level, address,
                                   'write to a read-only region')

    # ------------------------------------------------------------------
    # Memory attributes
    # ------------------------------------------------------------------

    def _memory_attributes(self, entry):
        """
        Decode the MAIR_EL1 byte the descriptor selected.

        Only the Normal/Device distinction is carried through, since that is what the
        rest of the model acts on; the cacheability hints are recorded but unused.
        """
        attributes = MemoryAttributes()
        field = substring(self.mair, 8 * entry.attr_index + 7, 8 * entry.attr_index)

        if substring(field, 7, 4) == 0:
            attributes.type = MemType.DEVICE
            attributes.innerattrs = 0
            attributes.outerattrs = 0
            # Device memory is always treated as outer shareable.
            attributes.shareable = True
            attributes.outershareable = True
        else:
            attributes.type = MemType.NORMAL
            attributes.outerattrs = substring(field, 7, 4)
            attributes.innerattrs = substring(field, 3, 0)
            attributes.shareable = bit_at(entry.shareability, 1) == 1
            attributes.outershareable = entry.shareability == 0b10
        return attributes

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def translate_stage2(self, ipa, is_write, is_instruction):
        """
        Translate an intermediate physical address through the stage 2 tables.

        Stage 2 is the hypervisor's translation: what a guest believes is physical memory
        is only an intermediate address, and this is the walk that turns it into a real
        one. The guest cannot see or change these tables, which is what makes the
        isolation hold.

        The descriptor format differs from stage 1 in the parts that matter here:
        permissions are a two-bit read/write field rather than the AP encoding, and
        there is no distinction between privileged and unprivileged access, because from
        stage 2's point of view the whole guest is one thing.
        """
        vtcr = self.vtcr
        tsz = substring(vtcr, 5, 0)
        granule = TG0_GRANULE.get(substring(vtcr, 15, 14))
        if granule is None:
            raise TranslationFault(FAULT_TRANSLATION, 0, ipa, 'reserved stage 2 granule')

        input_size = 64 - tsz
        if input_size < 64 and substring(ipa, 63, input_size) != 0:
            raise TranslationFault(FAULT_TRANSLATION, 0, ipa,
                                   'address outside the stage 2 region', stage2=True)

        stride = granule - 3
        # VTCR_EL2.SL0 names the starting level directly rather than leaving it to be
        # derived, because stage 2 tables may be concatenated at the top level.
        sl0 = substring(vtcr, 7, 6)
        first_level = {0b00: 2, 0b01: 1, 0b10: 0}.get(sl0)
        if first_level is None:
            raise TranslationFault(FAULT_TRANSLATION, 0, ipa, 'reserved SL0')

        level = first_level
        base_address = self.vttbr & ((1 << 48) - 1) & ~0xFFF
        first_index_bits = input_size - (granule + stride * (3 - first_level))

        while True:
            low_bit = granule + stride * (3 - level)
            index_bits = first_index_bits if level == first_level else stride
            index = substring(ipa, low_bit + index_bits - 1, low_bit)
            descriptor = self._read_descriptor(base_address + (index << 3))

            if not bit_at(descriptor, 0):
                raise TranslationFault(FAULT_TRANSLATION, level, ipa,
                                       'invalid stage 2 descriptor', stage2=True)

            if bit_at(descriptor, 1) and level < 3:
                base_address = substring(descriptor, 47, granule) << granule
                level += 1
                continue
            if not bit_at(descriptor, 1) and level == 3:
                raise TranslationFault(FAULT_TRANSLATION, level, ipa,
                                       'reserved stage 2 descriptor', stage2=True)

            if not bit_at(descriptor, 10):
                raise TranslationFault(FAULT_ACCESS_FLAG, level, ipa,
                                       'stage 2 access flag clear', stage2=True)

            # S2AP: bit 6 grants read, bit 7 grants write.
            readable = bit_at(descriptor, 6)
            writable = bit_at(descriptor, 7)
            execute_never = bit_at(descriptor, 54)
            if is_instruction and execute_never:
                raise TranslationFault(FAULT_PERMISSION, level, ipa,
                                       'stage 2 execute never', stage2=True)
            if is_write and not writable:
                raise TranslationFault(FAULT_PERMISSION, level, ipa,
                                       'stage 2 write to a read-only region', stage2=True)
            if not is_write and not readable:
                raise TranslationFault(FAULT_PERMISSION, level, ipa,
                                       'stage 2 region is not readable', stage2=True)

            output_base = substring(descriptor, 47, low_bit) << low_bit
            return output_base | lower_chunk(ipa, low_bit), substring(descriptor, 5, 2)

    def translate(self, address, is_write=False, is_instruction=False):
        """
        Translate ``address``, raising :class:`TranslationFault` on failure.
        """
        registers = self.processor.registers
        el = registers.current_el()

        high = bit_at(address, 55)
        if self._top_byte_ignored(high):
            # The top byte is available to software for tagging, so it must be stripped
            # before translation and sign-filled from bit 55.
            address = lower_chunk(address, 56)
            if high:
                address |= 0xFF << 56

        # Cached translations belong to a regime and a security state: the same virtual
        # address means different things at EL1, at EL2 and in the secure world.
        key = (registers.regime(), registers.secure, high, address >> 12)
        entry = self.tlb.get(key)
        if entry is None:
            entry = self.walk(address)
            # Cache against the 4KB page even when a larger block produced it.
            self.tlb[key] = TlbEntry(
                physical_address=entry.physical_address & ~0xFFF,
                ap=entry.ap, uxn=entry.uxn, pxn=entry.pxn, attr_index=entry.attr_index,
                nG=entry.nG, shareability=entry.shareability, ns=entry.ns, level=entry.level,
            )
            physical_address = entry.physical_address
        else:
            self.hits += 1
            physical_address = entry.physical_address | (address & 0xFFF)
        # The cached entry holds the stage 1 result. Stage 2 is applied below on every
        # access rather than being folded in, because its permissions are checked
        # separately and a hypervisor may change them without the guest knowing.

        self._check_permission(entry, address, is_write, is_instruction, el)

        attributes = self._memory_attributes(entry)
        if registers.stage2_enabled:
            # What stage 1 produced is only an intermediate address; the hypervisor's
            # tables decide where it really lands.
            try:
                physical_address, s2_attr = self.translate_stage2(
                    physical_address, is_write, is_instruction
                )
            except TranslationFault as fault:
                # The guest's virtual address is what the hypervisor needs to report in
                # FAR_EL2, with the intermediate address going to HPFAR_EL2.
                fault.intermediate_address = fault.address
                fault.address = address
                raise
            if substring(s2_attr, 3, 2) == 0:
                # Stage 2 can only make memory weaker, never stronger, so Device at
                # stage 2 wins over Normal at stage 1.
                attributes.type = MemType.DEVICE

        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = physical_address
        # The EL1&0 regime modelled here is always non-secure; the descriptor's NS bit
        # only carries meaning for a secure translation regime, which EL3 would own.
        # A translation made in secure state is a secure access unless the descriptor
        # says otherwise; in non-secure state everything is non-secure regardless.
        descriptor.paddress.ns = 0 if (registers.secure and not entry.ns) else 1
        descriptor.memattrs = attributes
        return descriptor
