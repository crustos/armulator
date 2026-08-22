"""
AArch64 register state.

Unlike AArch32 there is no register banking by processor mode: X0-X30 are flat. What is
banked is the stack pointer (one per exception level) and the exception-return state
(ELR/SPSR/ESR/FAR per EL). Register 31 is context dependent - it reads as zero in most
instructions (XZR) but denotes SP in a handful of them - so callers must say which they
mean rather than this class guessing.
"""

from armulator.armv8.bits_ops import (
    bit_at,
    chain,
    lower_chunk,
    set_bit_at,
    set_substring,
    substring,
)
from armulator.armv8.enums import EL, InstrSet

ZR = 31
LR = 30


class PSTATE:
    """
    Process state. This is architectural state held in discrete fields rather than a
    register - it is only assembled into a word when it is saved to an SPSR.
    """

    def __init__(self):
        self.n = 0
        self.z = 0
        self.c = 0
        self.v = 0
        self.d = 1  # Debug exception mask
        self.a = 1  # SError mask
        self.i = 1  # IRQ mask
        self.f = 1  # FIQ mask
        self.ss = 0  # Software step
        self.il = 0  # Illegal execution state
        self.el = EL.EL1
        self.n_rw = 0  # 0 = AArch64, 1 = AArch32
        self.sp = 1  # 0 = use SP_EL0, 1 = use SP_ELx

    @property
    def nzcv(self) -> int:
        return chain(chain(chain(self.n, self.z, 1), self.c, 1), self.v, 1)

    @nzcv.setter
    def nzcv(self, value: int) -> None:
        self.n = bit_at(value, 3)
        self.z = bit_at(value, 2)
        self.c = bit_at(value, 1)
        self.v = bit_at(value, 0)

    @property
    def daif(self) -> int:
        return chain(chain(chain(self.d, self.a, 1), self.i, 1), self.f, 1)

    @daif.setter
    def daif(self, value: int) -> None:
        self.d = bit_at(value, 3)
        self.a = bit_at(value, 2)
        self.i = bit_at(value, 1)
        self.f = bit_at(value, 0)

    def to_spsr(self) -> int:
        """
        Pack PSTATE into the SPSR_ELx layout used when taking an exception from AArch64.
        """
        value = 0
        value = set_substring(value, 31, 28, self.nzcv)
        value = set_bit_at(value, 21, self.ss)
        value = set_bit_at(value, 20, self.il)
        value = set_substring(value, 9, 6, self.daif)
        value = set_bit_at(value, 4, self.n_rw)
        value = set_substring(value, 3, 2, self.el.value)
        value = set_bit_at(value, 0, self.sp)
        return value

    def from_spsr(self, value: int) -> None:
        """
        Restore PSTATE from an SPSR on exception return.
        """
        self.nzcv = substring(value, 31, 28)
        self.ss = bit_at(value, 21)
        self.il = bit_at(value, 20)
        self.daif = substring(value, 9, 6)
        self.n_rw = bit_at(value, 4)
        self.el = EL(substring(value, 3, 2))
        self.sp = bit_at(value, 0)

    def __repr__(self):
        flags = ''.join(name if getattr(self, attr) else '-'
                        for name, attr in (('N', 'n'), ('Z', 'z'), ('C', 'c'), ('V', 'v')))
        masks = ''.join(name if getattr(self, attr) else '-'
                        for name, attr in (('D', 'd'), ('A', 'a'), ('I', 'i'), ('F', 'f')))
        return f'<PSTATE {flags} {masks} EL{self.el.value} SP{self.sp}>'


class Registers:
    def __init__(self):
        # X0-X30. X31 is never stored: it is XZR or SP depending on the instruction.
        self._x = [0] * 31
        # V0-V31, each 128 bits. Unlike the general purpose registers there is no
        # zero register here: V31 is an ordinary vector register.
        self._v = [0] * 32
        self._sp = {EL.EL0: 0, EL.EL1: 0, EL.EL2: 0, EL.EL3: 0}
        self._pc = 0
        self.pstate = PSTATE()

        # Exception return state, banked per exception level.
        self.elr = {EL.EL1: 0, EL.EL2: 0, EL.EL3: 0}
        self.spsr = {EL.EL1: 0, EL.EL2: 0, EL.EL3: 0}
        self.esr = {EL.EL1: 0, EL.EL2: 0, EL.EL3: 0}
        self.far = {EL.EL1: 0, EL.EL2: 0, EL.EL3: 0}
        self.vbar = {EL.EL1: 0, EL.EL2: 0, EL.EL3: 0}

        # Set when an instruction writes the PC, so the sequential increment is suppressed.
        self.branch_taken = False
        #: Loads whose value has not been settled yet, keyed by destination register.
        #: Each entry is (address, size, reload) - see ArmV8.note_deferred_load.
        self.pending_loads = {}
        self.event_register = False

        self.system_registers = {}
        self._init_system_registers()

    # ------------------------------------------------------------------
    # General purpose registers
    # ------------------------------------------------------------------

    def get_x(self, n: int, length: int = 64) -> int:
        """
        X[n] - register 31 reads as zero.

        A register holding an unresolved load is settled before it is read. That is what
        makes deferring a load safe: the program can never observe a provisional value,
        so a reordering is only ever visible when nothing depended on the result in the
        meantime - which is exactly when a real machine would have been free to reorder
        it too.
        """
        assert 0 <= n <= 31
        if n == ZR:
            return 0
        if n in self.pending_loads:
            self.resolve_pending_load(n)
        return lower_chunk(self._x[n], length)

    def resolve_pending_load(self, n) -> None:
        """Perform a deferred load now and put the result in its register."""
        entry = self.pending_loads.pop(n, None)
        if entry is None:
            return
        _, _, reload = entry
        self._x[n] = lower_chunk(reload(), 64)

    def resolve_all_pending_loads(self) -> None:
        for n in list(self.pending_loads):
            self.resolve_pending_load(n)

    def resolve_pending_loads_overlapping(self, address, size) -> None:
        """
        Settle any deferred load that touches a range about to be written.

        A load cannot be reordered past a store to the same location - the dependency
        is what stops it - so one that overlaps has to be performed before the store
        goes ahead.
        """
        if not self.pending_loads:
            return
        for n, (pending_address, pending_size, _) in list(self.pending_loads.items()):
            if pending_address < address + size and address < pending_address + pending_size:
                self.resolve_pending_load(n)

    def set_x(self, n: int, value: int, length: int = 64) -> None:
        """
        X[n] = value - writes to register 31 are discarded. A 32-bit write zeroes the
        upper half of the register, which is the single most common source of AArch64
        emulation bugs.
        """
        assert 0 <= n <= 31
        if n == ZR:
            return
        # Overwriting the destination discards any load still waiting to fill it.
        self.pending_loads.pop(n, None)
        self._x[n] = lower_chunk(value, length)

    # ------------------------------------------------------------------
    # SIMD and floating point registers
    # ------------------------------------------------------------------

    def get_v(self, n: int, length: int = 128) -> int:
        """
        Read the low ``length`` bits of Vn. The named views B/H/S/D/Q are just widths:
        S3 is the low 32 bits of V3.
        """
        assert 0 <= n <= 31
        return lower_chunk(self._v[n], length)

    def set_v(self, n: int, value: int, length: int = 128) -> None:
        """
        Write the low ``length`` bits of Vn, zeroing everything above.

        This zeroing is the SIMD counterpart of the 32-bit general purpose register rule
        and is just as easy to get wrong: writing D0 must clear bits 127:64 of V0, so a
        later read of Q0 sees zeros rather than stale data.
        """
        assert 0 <= n <= 31
        self._v[n] = lower_chunk(value, length)

    def get_v_element(self, n: int, index: int, element_size: int) -> int:
        """Read one lane of Vn, where ``element_size`` is in bits."""
        return substring(self._v[n], (index + 1) * element_size - 1, index * element_size)

    def set_v_element(self, n: int, index: int, element_size: int, value: int) -> None:
        """
        Write one lane of Vn, leaving the other lanes alone. Used by INS and by the
        by-element forms, which are the only SIMD writes that do not zero the rest.
        """
        high = (index + 1) * element_size - 1
        low = index * element_size
        self._v[n] = lower_chunk(
            set_substring(self._v[n], high, low, lower_chunk(value, element_size)), 128
        )

    def get_sp(self, length: int = 64) -> int:
        """
        SP[] - reads the stack pointer selected by PSTATE.SP and PSTATE.EL.
        """
        return lower_chunk(self._sp[self._selected_sp_el()], length)

    def set_sp(self, value: int, length: int = 64) -> None:
        self._sp[self._selected_sp_el()] = lower_chunk(value, length)

    def _selected_sp_el(self) -> EL:
        if self.pstate.sp == 0:
            return EL.EL0
        return self.pstate.el

    def get_sp_el(self, el: EL, length: int = 64) -> int:
        return lower_chunk(self._sp[el], length)

    def set_sp_el(self, el: EL, value: int, length: int = 64) -> None:
        self._sp[el] = lower_chunk(value, length)

    def get_reg_or_sp(self, n: int, length: int = 64) -> int:
        """
        For the instructions in which register 31 encodes SP rather than XZR
        (address operands, ADD/SUB immediate, and the extended-register forms).
        """
        if n == ZR:
            return self.get_sp(length)
        return self.get_x(n, length)

    def set_reg_or_sp(self, n: int, value: int, length: int = 64) -> None:
        if n == ZR:
            self.set_sp(value, length)
        else:
            self.set_x(n, value, length)

    def get_lr(self) -> int:
        return self.get_x(LR)

    def set_lr(self, value: int) -> None:
        self.set_x(LR, value)

    # ------------------------------------------------------------------
    # Program counter
    # ------------------------------------------------------------------

    def get_pc(self) -> int:
        """
        In AArch64 the PC reads as the address of the current instruction - there is no
        pipeline offset to account for as there was in AArch32.
        """
        return self._pc

    def branch_to(self, address: int) -> None:
        self._pc = lower_chunk(address, 64)
        self.branch_taken = True

    def increment_pc(self, length: int = 4) -> None:
        self._pc = lower_chunk(self._pc + length, 64)

    # ------------------------------------------------------------------
    # Execution state
    # ------------------------------------------------------------------

    def current_el(self) -> EL:
        return self.pstate.el

    def current_instr_set(self) -> InstrSet:
        return InstrSet.A32 if self.pstate.n_rw else InstrSet.A64

    def using_aarch32(self) -> bool:
        return bool(self.pstate.n_rw)

    def set_event_register(self, flag: bool) -> None:
        self.event_register = flag

    def get_event_register(self) -> bool:
        return self.event_register

    def condition_holds(self, cond: int) -> bool:
        """
        ConditionHolds(cond) - the 4-bit condition encoding shared by B.cond, CSEL and
        the conditional compares.
        """
        upper = substring(cond, 3, 1)
        if upper == 0b000:
            result = self.pstate.z == 1
        elif upper == 0b001:
            result = self.pstate.c == 1
        elif upper == 0b010:
            result = self.pstate.n == 1
        elif upper == 0b011:
            result = self.pstate.v == 1
        elif upper == 0b100:
            result = self.pstate.c == 1 and self.pstate.z == 0
        elif upper == 0b101:
            result = self.pstate.n == self.pstate.v
        elif upper == 0b110:
            result = self.pstate.n == self.pstate.v and self.pstate.z == 0
        else:
            result = True
        # The lowest bit inverts the condition, except for the "always" encoding.
        if bit_at(cond, 0) and cond != 0b1111:
            result = not result
        return result

    def set_flags(self, result: int, carry: int, overflow: int, length: int = 64) -> None:
        self.pstate.n = bit_at(result, length - 1)
        self.pstate.z = 1 if lower_chunk(result, length) == 0 else 0
        self.pstate.c = carry
        self.pstate.v = overflow

    # ------------------------------------------------------------------
    # System registers
    # ------------------------------------------------------------------

    def _init_system_registers(self) -> None:
        """
        System registers are keyed by their MSR/MRS encoding (op0, op1, CRn, CRm, op2),
        which is how the instruction addresses them. Reset values are those of a
        Cortex-A57 unless the register is architecturally UNKNOWN at reset.
        """
        self.system_registers = {
            # MIDR_EL1 - ARM, Cortex-A57
            (0b11, 0b000, 0b0000, 0b0000, 0b000): 0x411FD070,
            # MPIDR_EL1 - affinity 0, bit 31 RES1
            (0b11, 0b000, 0b0000, 0b0000, 0b101): 0x80000000,
            # ID_AA64PFR0_EL1 - EL0/EL1/EL2/EL3 in both states, AdvSIMD+FP
            (0b11, 0b000, 0b0000, 0b0100, 0b000): 0x0000000000002222,
            # ID_AA64MMFR0_EL1 - 40 bit PA, 4KB/64KB granules
            (0b11, 0b000, 0b0000, 0b0111, 0b000): 0x0000000000001122,
            # SCTLR_EL1 - RES1 bits set, MMU/caches off
            (0b11, 0b000, 0b0001, 0b0000, 0b000): 0x00C50838,
            # SCTLR_EL2 / SCTLR_EL3 - RES1 bits set, MMU off
            (0b11, 0b100, 0b0001, 0b0000, 0b000): 0x30C50830,
            (0b11, 0b110, 0b0001, 0b0000, 0b000): 0x30C50830,
            # HCR_EL2 - EL1 is AArch64 (RW), nothing trapped or virtualised yet
            (0b11, 0b100, 0b0001, 0b0001, 0b000): 1 << 31,
            # CPTR_EL2 / CPTR_EL3
            (0b11, 0b100, 0b0001, 0b0001, 0b010): 0,
            (0b11, 0b110, 0b0001, 0b0001, 0b010): 0,
            # SCR_EL3 - EL1/EL2 are AArch64 (RW); NS clear, so reset is to secure state
            (0b11, 0b110, 0b0001, 0b0001, 0b000): 1 << 10,
            # TTBR0_EL2 / TCR_EL2 / VTTBR_EL2 / VTCR_EL2
            (0b11, 0b100, 0b0010, 0b0000, 0b000): 0,
            (0b11, 0b100, 0b0010, 0b0000, 0b010): 0,
            (0b11, 0b100, 0b0010, 0b0001, 0b000): 0,
            (0b11, 0b100, 0b0010, 0b0001, 0b010): 0,
            # TTBR0_EL3 / TCR_EL3
            (0b11, 0b110, 0b0010, 0b0000, 0b000): 0,
            (0b11, 0b110, 0b0010, 0b0000, 0b010): 0,
            # MAIR_EL2 / MAIR_EL3
            (0b11, 0b100, 0b1010, 0b0010, 0b000): 0,
            (0b11, 0b110, 0b1010, 0b0010, 0b000): 0,
            # HPFAR_EL2 - the intermediate address of a stage 2 fault
            (0b11, 0b100, 0b0110, 0b0000, 0b100): 0,
            # CPACR_EL1
            (0b11, 0b000, 0b0001, 0b0000, 0b010): 0,
            # TTBR0_EL1 / TTBR1_EL1 / TCR_EL1
            (0b11, 0b000, 0b0010, 0b0000, 0b000): 0,
            (0b11, 0b000, 0b0010, 0b0000, 0b001): 0,
            (0b11, 0b000, 0b0010, 0b0000, 0b010): 0,
            # MAIR_EL1
            (0b11, 0b000, 0b1010, 0b0010, 0b000): 0,
            # TPIDR_EL0 / TPIDRRO_EL0 / TPIDR_EL1
            (0b11, 0b011, 0b1101, 0b0000, 0b010): 0,
            (0b11, 0b011, 0b1101, 0b0000, 0b011): 0,
            (0b11, 0b000, 0b1101, 0b0000, 0b100): 0,
            # FPCR / FPSR
            (0b11, 0b011, 0b0100, 0b0100, 0b000): 0,
            (0b11, 0b011, 0b0100, 0b0100, 0b001): 0,
            # CNTFRQ_EL0 - Jetson Nano runs its architected timer at 19.2MHz
            (0b11, 0b011, 0b1110, 0b0000, 0b000): 19200000,
        }

    def set_mpidr(self, cpu_id: int) -> None:
        """
        Set MPIDR_EL1 for this core.

        Aff0 holds the core number within the cluster and Aff1 the cluster number; a
        single Cortex-A57 cluster leaves Aff1 at zero. Bit 31 reads as one, and bit 30
        is clear because this is a multiprocessor implementation.
        """
        self.system_registers[(0b11, 0b000, 0b0000, 0b0000, 0b101)] = 0x80000000 | (cpu_id & 0xFF)

    def get_system_register(self, op0: int, op1: int, crn: int, crm: int, op2: int) -> int:
        """
        Read a system register. The registers that alias live PSTATE or banked state are
        handled specially so they never go stale against the fields they mirror.
        """
        key = (op0, op1, crn, crm, op2)
        special = self._special_system_register_read(key)
        if special is not None:
            return special
        return self.system_registers.get(key, 0)

    def set_system_register(self, op0: int, op1: int, crn: int, crm: int, op2: int, value: int) -> None:
        key = (op0, op1, crn, crm, op2)
        if self._special_system_register_write(key, value):
            return
        self.system_registers[key] = lower_chunk(value, 64)

    #: Registers that alias live state rather than sitting in the table, keyed by their
    #: MSR/MRS encoding. Each entry names which banked structure and which exception
    #: level it belongs to, so EL1, EL2 and EL3 are handled by one mechanism instead of
    #: three near-identical chains of comparisons.
    BANKED_REGISTERS = {
        # (op0, op1, CRn, CRm, op2): (bank, EL)
        (0b11, 0b000, 0b0100, 0b0000, 0b000): ('spsr', EL.EL1),
        (0b11, 0b000, 0b0100, 0b0000, 0b001): ('elr', EL.EL1),
        (0b11, 0b000, 0b0101, 0b0010, 0b000): ('esr', EL.EL1),
        (0b11, 0b000, 0b0110, 0b0000, 0b000): ('far', EL.EL1),
        (0b11, 0b000, 0b1100, 0b0000, 0b000): ('vbar', EL.EL1),

        (0b11, 0b100, 0b0100, 0b0000, 0b000): ('spsr', EL.EL2),
        (0b11, 0b100, 0b0100, 0b0000, 0b001): ('elr', EL.EL2),
        (0b11, 0b100, 0b0101, 0b0010, 0b000): ('esr', EL.EL2),
        (0b11, 0b100, 0b0110, 0b0000, 0b000): ('far', EL.EL2),
        (0b11, 0b100, 0b1100, 0b0000, 0b000): ('vbar', EL.EL2),

        (0b11, 0b110, 0b0100, 0b0000, 0b000): ('spsr', EL.EL3),
        (0b11, 0b110, 0b0100, 0b0000, 0b001): ('elr', EL.EL3),
        (0b11, 0b110, 0b0101, 0b0010, 0b000): ('esr', EL.EL3),
        (0b11, 0b110, 0b0110, 0b0000, 0b000): ('far', EL.EL3),
        (0b11, 0b110, 0b1100, 0b0000, 0b000): ('vbar', EL.EL3),
    }

    #: Stack pointers reachable by name from a higher exception level.
    BANKED_STACK_POINTERS = {
        (0b11, 0b000, 0b0100, 0b0001, 0b000): EL.EL0,
        (0b11, 0b100, 0b0100, 0b0001, 0b000): EL.EL1,
        (0b11, 0b110, 0b0100, 0b0001, 0b000): EL.EL2,
    }

    #: Encodings that only exist at or above a given exception level. Reading one from
    #: below is a trap on real hardware; here it simply reads as zero, which is enough
    #: to keep firmware from believing EL2 exists when it is running at EL1.
    MINIMUM_LEVEL = {0b100: EL.EL2, 0b110: EL.EL3}

    def _special_system_register_read(self, key):
        el = self.pstate.el
        if key == (0b11, 0b011, 0b0100, 0b0010, 0b000):  # NZCV
            return self.pstate.nzcv << 28
        if key == (0b11, 0b011, 0b0100, 0b0010, 0b001):  # DAIF
            return self.pstate.daif << 6
        if key == (0b11, 0b000, 0b0100, 0b0010, 0b010):  # CurrentEL
            return el.value << 2

        bank = self.BANKED_REGISTERS.get(key)
        if bank is not None:
            name, level = bank
            return getattr(self, name)[level]

        stack = self.BANKED_STACK_POINTERS.get(key)
        if stack is not None:
            return self.get_sp_el(stack)
        return None

    def _special_system_register_write(self, key, value) -> bool:
        if key == (0b11, 0b011, 0b0100, 0b0010, 0b000):  # NZCV
            self.pstate.nzcv = substring(value, 31, 28)
            return True
        if key == (0b11, 0b011, 0b0100, 0b0010, 0b001):  # DAIF
            self.pstate.daif = substring(value, 9, 6)
            return True
        if key == (0b11, 0b000, 0b0100, 0b0010, 0b010):  # CurrentEL is read only
            return True

        bank = self.BANKED_REGISTERS.get(key)
        if bank is not None:
            name, level = bank
            # SPSR and ESR are 32-bit; the address-holding ones are 64.
            width = 32 if name in ('spsr', 'esr') else 64
            getattr(self, name)[level] = lower_chunk(value, width)
            return True

        stack = self.BANKED_STACK_POINTERS.get(key)
        if stack is not None:
            self.set_sp_el(stack, value)
            return True
        return False

    # ------------------------------------------------------------------
    # Convenience accessors for the control registers used by the CPU model
    # ------------------------------------------------------------------

    #: Which system registers describe the translation regime of each exception level.
    #: EL0 has no regime of its own: it uses EL1's.
    REGIME_REGISTERS = {
        EL.EL1: {'sctlr': (0b11, 0b000, 0b0001, 0b0000, 0b000),
                 'tcr': (0b11, 0b000, 0b0010, 0b0000, 0b010),
                 'ttbr0': (0b11, 0b000, 0b0010, 0b0000, 0b000),
                 'ttbr1': (0b11, 0b000, 0b0010, 0b0000, 0b001),
                 'mair': (0b11, 0b000, 0b1010, 0b0010, 0b000)},
        EL.EL2: {'sctlr': (0b11, 0b100, 0b0001, 0b0000, 0b000),
                 'tcr': (0b11, 0b100, 0b0010, 0b0000, 0b010),
                 'ttbr0': (0b11, 0b100, 0b0010, 0b0000, 0b000),
                 'ttbr1': None,
                 'mair': (0b11, 0b100, 0b1010, 0b0010, 0b000)},
        EL.EL3: {'sctlr': (0b11, 0b110, 0b0001, 0b0000, 0b000),
                 'tcr': (0b11, 0b110, 0b0010, 0b0000, 0b010),
                 'ttbr0': (0b11, 0b110, 0b0010, 0b0000, 0b000),
                 'ttbr1': None,
                 'mair': (0b11, 0b110, 0b1010, 0b0010, 0b000)},
    }

    def regime(self) -> EL:
        """
        Which translation regime is in force. EL0 translates using EL1's tables, which
        is why an application and its kernel share a page table hierarchy.
        """
        el = self.pstate.el
        return EL.EL1 if el == EL.EL0 else el

    def regime_register(self, name, el=None) -> int:
        """Read one of the translation control registers for a regime."""
        key = self.REGIME_REGISTERS[el or self.regime()][name]
        if key is None:
            return 0
        return self.get_system_register(*key)

    @property
    def hcr_el2(self) -> int:
        return self.get_system_register(0b11, 0b100, 0b0001, 0b0001, 0b000)

    @property
    def scr_el3(self) -> int:
        return self.get_system_register(0b11, 0b110, 0b0001, 0b0001, 0b000)

    @property
    def stage2_enabled(self) -> bool:
        """
        HCR_EL2.VM turns on stage 2 translation for EL0 and EL1, which is what makes a
        guest's idea of physical memory the hypervisor's idea of an intermediate address.
        """
        return bool(bit_at(self.hcr_el2, 0)) and self.pstate.el in (EL.EL0, EL.EL1)

    @property
    def secure(self) -> bool:
        """
        Whether execution is in secure state. EL3 is always secure; below it, SCR_EL3.NS
        decides. Reset is to secure state at the highest implemented level.
        """
        if self.pstate.el == EL.EL3:
            return True
        return not bit_at(self.scr_el3, 0)

    @property
    def sctlr_el1(self) -> int:
        return self.system_registers[(0b11, 0b000, 0b0001, 0b0000, 0b000)]

    @sctlr_el1.setter
    def sctlr_el1(self, value: int) -> None:
        self.system_registers[(0b11, 0b000, 0b0001, 0b0000, 0b000)] = lower_chunk(value, 64)

    @property
    def mmu_enabled(self) -> bool:
        """Whether stage 1 translation is on for the current regime."""
        return bool(bit_at(self.regime_register('sctlr'), 0))

    @property
    def stack_alignment_check(self) -> bool:
        return bool(bit_at(self.regime_register('sctlr'), 3))  # SCTLR.SA

    @property
    def alignment_check(self) -> bool:
        return bool(bit_at(self.regime_register('sctlr'), 1))  # SCTLR.A

    @property
    def fpcr(self) -> int:
        return self.system_registers.get((0b11, 0b011, 0b0100, 0b0100, 0b000), 0)

    @fpcr.setter
    def fpcr(self, value: int) -> None:
        self.system_registers[(0b11, 0b011, 0b0100, 0b0100, 0b000)] = lower_chunk(value, 64)

    @property
    def fpsr(self) -> int:
        return self.system_registers.get((0b11, 0b011, 0b0100, 0b0100, 0b001), 0)

    @fpsr.setter
    def fpsr(self, value: int) -> None:
        self.system_registers[(0b11, 0b011, 0b0100, 0b0100, 0b001)] = lower_chunk(value, 64)

    @property
    def fp_rounding_mode(self) -> int:
        """FPCR.RMode - the rounding mode used by most floating point operations."""
        return substring(self.fpcr, 23, 22)

    @property
    def fp_access_enabled(self) -> bool:
        """
        CPACR_EL1.FPEN must be 11 (or 01 outside EL0) for SIMD and floating point to be
        usable. It resets to 00, so startup code has to enable FP explicitly before any
        vector instruction will execute - exactly as on real hardware.
        """
        fpen = substring(self.system_registers.get((0b11, 0b000, 0b0001, 0b0000, 0b010), 0), 21, 20)
        if fpen == 0b11:
            return True
        if fpen == 0b01:
            return self.pstate.el != EL.EL0
        return False

    def reset(self) -> None:
        """
        Take reset - AArch64 at EL1 with SP_EL1 selected and all exceptions masked.
        """
        self._x = [0] * 31
        self._v = [0] * 32
        self.pending_loads = {}
        self._sp = {EL.EL0: 0, EL.EL1: 0, EL.EL2: 0, EL.EL3: 0}
        self._pc = 0
        self.pstate = PSTATE()
        self.branch_taken = False
        self._init_system_registers()

    def format_registers(self) -> str:
        lines = []
        for i in range(0, 31, 2):
            left = f'X{i}:'.ljust(5) + f'0x{self.get_x(i):016X}'
            if i + 1 < 31:
                right = f'X{i + 1}:'.ljust(5) + f'0x{self.get_x(i + 1):016X}'
                lines.append(f'{left}   {right}')
            else:
                lines.append(left)
        lines.append(f'SP:  0x{self.get_sp():016X}   PC:  0x{self.get_pc():016X}')
        lines.append(str(self.pstate))
        return '\n'.join(lines)
