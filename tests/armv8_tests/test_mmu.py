"""
Stage 1 address translation.

Each test builds real page tables in emulator memory and translates through them, so the
descriptor encodings are exercised rather than mocked.
"""

import pytest

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.armv6.memory_attributes import MemType
from armulator.armv8.arm_exceptions import DataAbortException, InstructionAbortException
from armulator.armv8.arm_v8 import ArmV8
from armulator.armv8.enums import EL

RAM = [{'mem_type': 'RAM', 'beginning': 0x0, 'end': 0x800000}]

VALID = 1
TABLE = 2
ACCESS_FLAG = 1 << 10
INNER_SHAREABLE = 3 << 8
UXN = 1 << 54
PXN = 1 << 53

#: AP[2:1] encodings, as they sit in the descriptor.
AP_EL1_RW = 0b00 << 6
AP_EL0_RW = 0b01 << 6
AP_EL1_RO = 0b10 << 6
AP_EL0_RO = 0b11 << 6

#: MAIR indices used by the fixtures below.
ATTR_NORMAL = 0 << 2
ATTR_DEVICE = 1 << 2

LEAF = TABLE | VALID | ACCESS_FLAG | INNER_SHAREABLE


class PageTables:
    """
    A small 4KB-granule table builder writing straight into emulator memory.
    """

    def __init__(self, processor, base=0x100000, table_size=0x1000):
        # Tables are aligned to the granule size, so a 64KB granule needs 64KB tables
        # rather than 4KB ones - the walk masks the descriptor's address to the granule.
        self.processor = processor
        self.table_size = table_size
        self.next_free = base
        self.root = self.allocate()

    def allocate(self):
        table = self.next_free
        self.next_free += self.table_size
        for offset in range(0, self.table_size, 8):
            self.write(table + offset, 0)
        return table

    def write(self, address, value):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address
        self.processor.mem[descriptor, 8] = value

    def read(self, address):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address
        return self.processor.mem[descriptor, 8]

    def _descend(self, table, index):
        existing = self.read(table + index * 8)
        if existing & VALID:
            return existing & 0x0000FFFFFFFFF000
        child = self.allocate()
        self.write(table + index * 8, child | TABLE | VALID)
        return child

    def map_page(self, va, pa, flags=AP_EL1_RW | ATTR_NORMAL):
        """Map one 4KB page, creating tables as needed."""
        l1 = self._descend(self.root, (va >> 39) & 0x1FF)
        l2 = self._descend(l1, (va >> 30) & 0x1FF)
        l3 = self._descend(l2, (va >> 21) & 0x1FF)
        self.write(l3 + ((va >> 12) & 0x1FF) * 8, pa | LEAF | flags)

    def map_block_2mb(self, va, pa, flags=AP_EL1_RW | ATTR_NORMAL):
        l1 = self._descend(self.root, (va >> 39) & 0x1FF)
        l2 = self._descend(l1, (va >> 30) & 0x1FF)
        # A block descriptor has bits[1:0] = 01, so TABLE is deliberately absent.
        self.write(l2 + ((va >> 21) & 0x1FF) * 8,
                   pa | VALID | ACCESS_FLAG | INNER_SHAREABLE | flags)

    def map_block_1gb(self, va, pa, flags=AP_EL1_RW | ATTR_NORMAL):
        l1 = self._descend(self.root, (va >> 39) & 0x1FF)
        self.write(l1 + ((va >> 30) & 0x1FF) * 8,
                   pa | VALID | ACCESS_FLAG | INNER_SHAREABLE | flags)

    def raw_leaf(self, va, value):
        """Write an arbitrary level 3 descriptor, for the malformed cases."""
        l1 = self._descend(self.root, (va >> 39) & 0x1FF)
        l2 = self._descend(l1, (va >> 30) & 0x1FF)
        l3 = self._descend(l2, (va >> 21) & 0x1FF)
        self.write(l3 + ((va >> 12) & 0x1FF) * 8, value)


def make_processor(root=None, t0sz=16, granule_tg0=0b00, ttbr1=None):
    processor = ArmV8(RAM)
    processor.take_reset()
    registers = processor.registers
    # attr0 Normal write-back, attr1 Device-nGnRnE
    registers.set_system_register(0b11, 0b000, 0b1010, 0b0010, 0b000, 0xFF | (0x00 << 8))
    tcr = t0sz | (t0sz << 16) | (granule_tg0 << 14) | (0b10 << 30)
    registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b010, tcr)
    if root is not None:
        registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, root)
        registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b001,
                                      ttbr1 if ttbr1 is not None else root)
    return processor


def enable_mmu(processor):
    processor.registers.sctlr_el1 |= 1
    processor.mmu.flush()


def write_physical(processor, address, value, size=8):
    descriptor = AddressDescriptor()
    descriptor.paddress.physicaladdress = address
    processor.mem[descriptor, size] = value


@pytest.fixture
def mapped():
    """A processor with 0x40000000 mapped to 0x200000 and the MMU on."""
    processor = make_processor()
    tables = PageTables(processor)
    tables.map_page(0x40000000, 0x200000, AP_EL0_RW | ATTR_NORMAL)
    processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, tables.root)
    processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b001, tables.root)
    enable_mmu(processor)
    return processor, tables


class TestBasicTranslation:
    def test_page_maps_to_its_physical_frame(self, mapped):
        processor, _ = mapped
        assert processor.translate_address(0x40000000).paddress.physicaladdress == 0x200000

    def test_offset_within_the_page_is_preserved(self, mapped):
        processor, _ = mapped
        assert processor.translate_address(0x40000ABC).paddress.physicaladdress == 0x200ABC

    def test_data_read_and_write_go_through_translation(self, mapped):
        processor, _ = mapped
        write_physical(processor, 0x200000, 0xDEAD)
        assert processor.mem_get(0x40000000, 8) == 0xDEAD
        processor.mem_set(0x40000000, 8, 0xBEEF)
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = 0x200000
        assert processor.mem[descriptor, 8] == 0xBEEF

    def test_mmu_off_is_flat(self):
        processor = make_processor()
        assert processor.translate_address(0x40000000).paddress.physicaladdress == 0x40000000


class TestBlockDescriptors:
    def test_2mb_block(self):
        processor = make_processor()
        tables = PageTables(processor)
        tables.map_block_2mb(0x40000000, 0x400000)
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, tables.root)
        enable_mmu(processor)
        assert processor.translate_address(0x40000000).paddress.physicaladdress == 0x400000
        # The block covers 2MB, so an address near its top still resolves.
        assert processor.translate_address(0x401FF000).paddress.physicaladdress == 0x5FF000

    def test_1gb_block(self):
        processor = make_processor()
        tables = PageTables(processor)
        tables.map_block_1gb(0x00000000, 0x00000000)
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, tables.root)
        enable_mmu(processor)
        assert processor.translate_address(0x12345).paddress.physicaladdress == 0x12345


class TestFaults:
    def _fault(self, processor, va, **kwargs):
        with pytest.raises(DataAbortException) as caught:
            processor.translate_address(va, **kwargs)
        return caught.value

    def test_unmapped_address_is_a_translation_fault(self, mapped):
        processor, _ = mapped
        fault = self._fault(processor, 0x50000000)
        # DFSC 0b0001LL: translation fault, with the level in the low two bits.
        assert fault.status >> 2 == 0b0001

    def test_write_to_read_only_page(self):
        processor = make_processor()
        tables = PageTables(processor)
        tables.map_page(0x40000000, 0x200000, AP_EL1_RO)
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, tables.root)
        enable_mmu(processor)
        assert processor.translate_address(0x40000000).paddress.physicaladdress == 0x200000
        fault = self._fault(processor, 0x40000000, is_write=True)
        assert fault.status == 0b001111        # permission fault, level 3

    def test_el0_cannot_reach_an_el1_only_page(self, mapped):
        processor, tables = mapped
        tables.map_page(0x40001000, 0x201000, AP_EL1_RW)
        processor.mmu.flush()
        assert processor.translate_address(0x40001000).paddress.physicaladdress == 0x201000
        processor.registers.pstate.el = EL.EL0
        fault = self._fault(processor, 0x40001000)
        assert fault.status >> 2 == 0b0011     # permission fault

    def test_clear_access_flag_faults(self):
        processor = make_processor()
        tables = PageTables(processor)
        # A leaf without the access flag: software has to set it, so this is a fault.
        tables.raw_leaf(0x40000000, 0x200000 | TABLE | VALID | INNER_SHAREABLE)
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, tables.root)
        enable_mmu(processor)
        fault = self._fault(processor, 0x40000000)
        assert fault.status == 0b001011        # access flag fault, level 3

    def test_fault_records_the_faulting_address(self, mapped):
        processor, _ = mapped
        self._fault(processor, 0x50000000)
        assert processor.registers.far[EL.EL1] == 0x50000000

    def test_data_abort_syndrome_reaches_esr(self, mapped):
        processor, _ = mapped
        processor.registers.vbar[EL.EL1] = 0x2000
        processor.registers.branch_to(0x40000000)
        # Execute a store to an unmapped address and let the CPU vector.
        processor.mem_set(0x40000000, 4, 0xB9000000)   # str w0, [x0]
        processor.registers.set_x(0, 0x50000000)
        processor.registers.branch_to(0x40000000)
        processor.registers.branch_taken = False
        processor.emulate_cycle()
        esr = processor.registers.esr[EL.EL1]
        assert esr >> 26 == 0b100101           # data abort, current EL
        assert (esr >> 6) & 1 == 1             # WnR: it was a write
        assert esr & 0b111111 == 0b000110      # translation fault, level 2


class TestExecutePermissions:
    def _processor_with(self, flags):
        processor = make_processor()
        tables = PageTables(processor)
        tables.map_page(0x40000000, 0x200000, flags)
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, tables.root)
        enable_mmu(processor)
        return processor

    def test_pxn_blocks_fetch_at_el1_but_not_data(self):
        processor = self._processor_with(AP_EL1_RW | PXN)
        assert processor.translate_address(0x40000000).paddress.physicaladdress == 0x200000
        with pytest.raises(InstructionAbortException) as caught:
            processor.translate_address(0x40000000, is_instruction=True)
        assert caught.value.status >> 2 == 0b0011

    def test_uxn_blocks_fetch_at_el0_only(self):
        processor = self._processor_with(AP_EL0_RW | UXN)
        # EL1 may still execute, since UXN governs EL0.
        processor.translate_address(0x40000000, is_instruction=True)
        processor.registers.pstate.el = EL.EL0
        with pytest.raises(InstructionAbortException):
            processor.translate_address(0x40000000, is_instruction=True)


class TestTtbrSelection:
    def test_high_addresses_use_ttbr1(self):
        processor = make_processor()
        low = PageTables(processor, base=0x100000)
        high = PageTables(processor, base=0x180000)
        kernel_va = 0xFFFF000000000000
        high.map_page(kernel_va, 0x300000, AP_EL1_RW)
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, low.root)
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b001, high.root)
        enable_mmu(processor)
        assert processor.translate_address(kernel_va).paddress.physicaladdress == 0x300000

    def test_the_middle_of_the_address_space_is_unmapped(self, mapped):
        processor, _ = mapped
        # Neither all-zeros nor all-ones on top, so neither base register applies.
        with pytest.raises(DataAbortException):
            processor.translate_address(0x0000800000000000)


class TestMemoryAttributes:
    def test_device_and_normal_come_from_mair(self):
        processor = make_processor()
        tables = PageTables(processor)
        tables.map_page(0x40000000, 0x200000, AP_EL1_RW | ATTR_NORMAL)
        tables.map_page(0x40001000, 0x201000, AP_EL1_RW | ATTR_DEVICE)
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, tables.root)
        enable_mmu(processor)
        assert processor.translate_address(0x40000000).memattrs.type == MemType.NORMAL
        assert processor.translate_address(0x40001000).memattrs.type == MemType.DEVICE


class TestTableDescriptorPermissions:
    def test_aptable_read_only_overrides_a_writable_leaf(self):
        processor = make_processor()
        tables = PageTables(processor)
        tables.map_page(0x40000000, 0x200000, AP_EL1_RW)
        # Set APTable[0] on the level 2 descriptor: everything below becomes read-only,
        # even though the leaf itself says read/write.
        l1 = tables._descend(tables.root, 0)
        l2 = tables._descend(l1, 1)
        entry_address = l2 + 0 * 8
        tables.write(entry_address, tables.read(entry_address) | (1 << 61))
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, tables.root)
        enable_mmu(processor)
        processor.translate_address(0x40000000)
        with pytest.raises(DataAbortException):
            processor.translate_address(0x40000000, is_write=True)


class TestGranules:
    def test_64kb_granule(self):
        processor = make_processor(granule_tg0=0b01)
        tables = PageTables(processor, table_size=0x10000)
        # 64KB granule: level 1 resolves VA[47:42], level 2 [41:29], level 3 [28:16].
        va = 0x40000000
        l2 = tables.allocate()
        l3 = tables.allocate()
        tables.write(tables.root + ((va >> 42) & 0x3F) * 8, l2 | TABLE | VALID)
        tables.write(l2 + ((va >> 29) & 0x1FFF) * 8, l3 | TABLE | VALID)
        tables.write(l3 + ((va >> 16) & 0x1FFF) * 8, 0x400000 | LEAF | AP_EL1_RW)
        processor.registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, tables.root)
        enable_mmu(processor)
        assert processor.translate_address(va).paddress.physicaladdress == 0x400000
        # The page is 64KB, so a large offset still lands inside it.
        assert processor.translate_address(va + 0x1234).paddress.physicaladdress == 0x401234


class TestTlb:
    def test_repeated_access_hits_the_cache(self, mapped):
        processor, _ = mapped
        processor.mmu.flush()
        processor.translate_address(0x40000000)
        walks = processor.mmu.walks
        for _ in range(5):
            processor.translate_address(0x40000000)
        assert processor.mmu.walks == walks
        assert processor.mmu.hits >= 5

    def test_stale_entry_persists_until_invalidated(self, mapped):
        # Editing a table without a TLBI leaves the old translation live, which is what
        # real hardware does and why firmware must invalidate.
        processor, tables = mapped
        processor.translate_address(0x40000000)
        tables.map_page(0x40000000, 0x300000, AP_EL0_RW)
        assert processor.translate_address(0x40000000).paddress.physicaladdress == 0x200000
        processor.mmu.flush()
        assert processor.translate_address(0x40000000).paddress.physicaladdress == 0x300000

    def test_writing_ttbr0_invalidates(self, mapped):
        processor, _ = mapped
        processor.translate_address(0x40000000)
        assert processor.mmu.tlb
        # Going through the instruction path exercises the flush hook on MSR.
        processor.registers.branch_to(0x40000000)
        processor.mem_set(0x40000000, 4, 0xD5182000)   # msr ttbr0_el1, x0
        processor.registers.set_x(0, processor.registers.get_system_register(
            0b11, 0b000, 0b0010, 0b0000, 0b000))
        processor.registers.branch_to(0x40000000)
        processor.registers.branch_taken = False
        processor.emulate_cycle()
        assert processor.mmu.tlb == {}
