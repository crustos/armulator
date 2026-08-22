from armulator.armv8.enums import EL
from armulator.armv8.registers import PSTATE, Registers


class TestGeneralPurposeRegisters:
    def test_register_31_reads_as_zero(self):
        regs = Registers()
        regs.set_x(31, 0xDEADBEEF)
        assert regs.get_x(31) == 0

    def test_32_bit_write_zeroes_upper_half(self):
        regs = Registers()
        regs.set_x(0, 0xFFFFFFFFFFFFFFFF)
        regs.set_x(0, 0x12345678, 32)
        assert regs.get_x(0) == 0x0000000012345678

    def test_register_31_is_sp_in_sp_form(self):
        regs = Registers()
        regs.set_reg_or_sp(31, 0x8000)
        assert regs.get_reg_or_sp(31) == 0x8000
        assert regs.get_x(31) == 0


class TestStackPointerSelection:
    def test_sp_follows_pstate_sp_and_el(self):
        regs = Registers()
        regs.set_sp_el(EL.EL0, 0x1000)
        regs.set_sp_el(EL.EL1, 0x2000)

        regs.pstate.el = EL.EL1
        regs.pstate.sp = 1
        assert regs.get_sp() == 0x2000

        regs.pstate.sp = 0
        assert regs.get_sp() == 0x1000


class TestPstate:
    def test_spsr_round_trip(self):
        pstate = PSTATE()
        pstate.n, pstate.z, pstate.c, pstate.v = 1, 0, 1, 0
        pstate.el = EL.EL1
        pstate.sp = 1
        pstate.d = pstate.a = pstate.i = pstate.f = 1

        packed = pstate.to_spsr()
        restored = PSTATE()
        restored.from_spsr(packed)

        assert (restored.n, restored.z, restored.c, restored.v) == (1, 0, 1, 0)
        assert restored.el == EL.EL1
        assert restored.sp == 1
        assert restored.daif == 0b1111

    def test_nzcv_packing(self):
        pstate = PSTATE()
        pstate.nzcv = 0b1010
        assert (pstate.n, pstate.z, pstate.c, pstate.v) == (1, 0, 1, 0)
        assert pstate.nzcv == 0b1010


class TestConditionHolds:
    def test_eq_and_ne(self):
        regs = Registers()
        regs.pstate.z = 1
        assert regs.condition_holds(0b0000) is True    # EQ
        assert regs.condition_holds(0b0001) is False   # NE

    def test_always(self):
        regs = Registers()
        assert regs.condition_holds(0b1110) is True    # AL
        assert regs.condition_holds(0b1111) is True    # NV behaves as always

    def test_signed_greater_than(self):
        regs = Registers()
        regs.pstate.n = regs.pstate.v = 0
        regs.pstate.z = 0
        assert regs.condition_holds(0b1100) is True    # GT


class TestSystemRegisters:
    def test_midr_identifies_cortex_a57(self):
        regs = Registers()
        midr = regs.get_system_register(0b11, 0b000, 0b0000, 0b0000, 0b000)
        # Implementer 0x41 (ARM), part number 0xD07 (Cortex-A57)
        assert (midr >> 24) & 0xFF == 0x41
        assert (midr >> 4) & 0xFFF == 0xD07

    def test_nzcv_aliases_live_pstate(self):
        regs = Registers()
        regs.pstate.nzcv = 0b1100
        assert regs.get_system_register(0b11, 0b011, 0b0100, 0b0010, 0b000) >> 28 == 0b1100
        regs.set_system_register(0b11, 0b011, 0b0100, 0b0010, 0b000, 0b0011 << 28)
        assert regs.pstate.nzcv == 0b0011

    def test_current_el_is_read_only(self):
        regs = Registers()
        regs.pstate.el = EL.EL1
        key = (0b11, 0b000, 0b0100, 0b0010, 0b010)
        regs.set_system_register(*key, 0b11 << 2)
        assert regs.get_system_register(*key) >> 2 == 0b01
