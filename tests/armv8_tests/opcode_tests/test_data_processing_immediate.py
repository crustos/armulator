"""
Data processing (immediate) group. Sources are assembled with keystone so the tests run
against the real encodings rather than against hand-built words.
"""


class TestMoveWide:
    def test_movz_with_shift(self, run_a64):
        proc = run_a64('movz x0, #0x1234, lsl #16')
        assert proc.registers.get_x(0) == 0x0000000012340000

    def test_movk_merges_into_slice(self, run_a64):
        proc = run_a64('movz x0, #0x1234, lsl #16\n movk x0, #0xABCD')
        assert proc.registers.get_x(0) == 0x000000001234ABCD

    def test_movn_inverts(self, run_a64):
        proc = run_a64('movn x0, #0')
        assert proc.registers.get_x(0) == 0xFFFFFFFFFFFFFFFF

    def test_movz_32_bit_zeroes_upper_half(self, run_a64):
        proc = run_a64('movn x0, #0\n movz w0, #1')
        assert proc.registers.get_x(0) == 0x0000000000000001


class TestAddSubImmediate:
    def test_add_and_sub(self, run_a64):
        proc = run_a64('movz x0, #100\n add x1, x0, #23\n sub x2, x1, #123')
        assert proc.registers.get_x(1) == 123
        assert proc.registers.get_x(2) == 0

    def test_shifted_immediate(self, run_a64):
        proc = run_a64('movz x0, #1\n add x1, x0, #1, lsl #12')
        assert proc.registers.get_x(1) == 0x1001

    def test_cmp_equal_sets_z_and_c(self, run_a64):
        proc = run_a64('movz x0, #5\n cmp x0, #5')
        assert proc.registers.pstate.z == 1
        # Subtraction sets carry as "not borrow", so an equal compare sets C.
        assert proc.registers.pstate.c == 1
        assert proc.registers.pstate.n == 0
        assert proc.registers.pstate.v == 0

    def test_cmp_less_than_clears_carry(self, run_a64):
        proc = run_a64('movz x0, #1\n cmp x0, #5')
        assert proc.registers.pstate.c == 0
        assert proc.registers.pstate.n == 1

    def test_subs_overflow(self, run_a64):
        # 0x8000000000000000 - 1 overflows the signed range.
        proc = run_a64('movz x0, #1, lsl #48\n lsl x0, x0, #15\n subs x1, x0, #1')
        assert proc.registers.get_x(0) == 0x8000000000000000
        assert proc.registers.pstate.v == 1

    def test_add_writes_sp(self, run_a64):
        proc = run_a64('movz x0, #0x2000\n mov sp, x0\n add sp, sp, #16')
        assert proc.registers.get_sp() == 0x2010

    def test_32_bit_add_zeroes_upper_half(self, run_a64):
        proc = run_a64('movn x0, #0\n add w1, w0, #0')
        assert proc.registers.get_x(1) == 0x00000000FFFFFFFF


class TestLogicalImmediate:
    def test_and_masks(self, run_a64):
        proc = run_a64('movn x0, #0\n and x1, x0, #0xff')
        assert proc.registers.get_x(1) == 0xFF

    def test_orr_from_zero_register(self, run_a64):
        proc = run_a64('orr x0, xzr, #0x1')
        assert proc.registers.get_x(0) == 1

    def test_eor(self, run_a64):
        proc = run_a64('movn x0, #0\n eor x1, x0, #0xff')
        assert proc.registers.get_x(1) == 0xFFFFFFFFFFFFFF00

    def test_ands_sets_flags_and_clears_c_v(self, run_a64):
        proc = run_a64('movz x0, #0\n ands x1, x0, #0xff')
        assert proc.registers.pstate.z == 1
        assert proc.registers.pstate.c == 0
        assert proc.registers.pstate.v == 0

    def test_replicated_pattern(self, run_a64):
        proc = run_a64('movn x0, #0\n and x1, x0, #0x5555555555555555')
        assert proc.registers.get_x(1) == 0x5555555555555555


class TestBitfield:
    def test_ubfx_zero_extends(self, run_a64):
        proc = run_a64('movz x0, #0xFF00\n ubfx x1, x0, #8, #8')
        assert proc.registers.get_x(1) == 0xFF

    def test_sbfx_sign_extends(self, run_a64):
        proc = run_a64('movz x0, #0xFF00\n sbfx x1, x0, #8, #8')
        assert proc.registers.get_x(1) == 0xFFFFFFFFFFFFFFFF

    def test_lsl_alias(self, run_a64):
        proc = run_a64('movz x0, #1\n lsl x1, x0, #4')
        assert proc.registers.get_x(1) == 0x10

    def test_lsr_alias(self, run_a64):
        proc = run_a64('movn x0, #0\n lsr x1, x0, #60')
        assert proc.registers.get_x(1) == 0xF

    def test_asr_alias(self, run_a64):
        proc = run_a64('movn x0, #0\n asr x1, x0, #60')
        assert proc.registers.get_x(1) == 0xFFFFFFFFFFFFFFFF

    def test_sxtb_alias(self, run_a64):
        proc = run_a64('movz x0, #0x80\n sxtb x1, w0')
        assert proc.registers.get_x(1) == 0xFFFFFFFFFFFFFF80

    def test_uxth_alias(self, run_a64):
        proc = run_a64('movn x0, #0\n uxth w1, w0')
        assert proc.registers.get_x(1) == 0xFFFF

    def test_bfi_preserves_surrounding_bits(self, run_a64):
        # BFM must leave the bits outside the inserted field untouched.
        proc = run_a64('movn x0, #0\n movz x1, #0\n bfi x1, x0, #8, #8')
        assert proc.registers.get_x(1) == 0xFF00


class TestExtract:
    def test_extr_concatenates(self, run_a64):
        proc = run_a64('movz x0, #0\n movn x1, #0\n extr x2, x0, x1, #32')
        assert proc.registers.get_x(2) == 0x00000000FFFFFFFF

    def test_ror_alias(self, run_a64):
        proc = run_a64('movz x0, #0xF\n ror x1, x0, #4')
        assert proc.registers.get_x(1) == 0xF000000000000000


class TestPcRelativeAddressing:
    def test_adr_reaches_a_label(self, run_a64):
        proc = run_a64('adr x0, target\n nop\n target: nop')
        assert proc.registers.get_x(0) == 0x1008

    def test_adrp_aligns_to_page(self, run_a64):
        # Assembled at 0x1000, so the PC page is 0x1000 and a zero offset stays there.
        proc = run_a64('adrp x0, #0x1000')
        assert proc.registers.get_x(0) == 0x1000
