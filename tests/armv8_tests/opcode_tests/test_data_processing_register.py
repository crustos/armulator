"""
Data processing (register).
"""


class TestAddSubShiftedRegister:
    def test_add_and_sub(self, run_a64):
        proc = run_a64('movz x1, #10\n movz x2, #3\n add x0, x1, x2\n sub x3, x1, x2')
        assert proc.registers.get_x(0) == 13
        assert proc.registers.get_x(3) == 7

    def test_shifted_operand(self, run_a64):
        proc = run_a64('movz x0, #1\n add x1, x0, x0, lsl #4')
        assert proc.registers.get_x(1) == 17

    def test_cmp_sets_flags(self, run_a64):
        proc = run_a64('movz x0, #5\n movz x1, #5\n cmp x0, x1')
        assert proc.registers.pstate.z == 1
        assert proc.registers.pstate.c == 1

    def test_neg_is_sub_from_zero(self, run_a64):
        proc = run_a64('movz x0, #1\n neg x1, x0')
        assert proc.registers.get_x(1) == 0xFFFFFFFFFFFFFFFF

    def test_32_bit_form_zeroes_upper_half(self, run_a64):
        proc = run_a64('movn x0, #0\n movz w1, #0\n add w2, w0, w1')
        assert proc.registers.get_x(2) == 0x00000000FFFFFFFF


class TestAddSubExtendedRegister:
    def test_extended_add_with_sp(self, run_a64):
        proc = run_a64('movz x0, #0x2000\n mov sp, x0\n movz x1, #0x10\n add x2, sp, x1')
        assert proc.registers.get_x(2) == 0x2010

    def test_byte_extension(self, run_a64):
        proc = run_a64('movz x0, #0x100\n movn x1, #0\n add x2, x0, w1, uxtb')
        # Only the low byte of the extended operand contributes.
        assert proc.registers.get_x(2) == 0x1FF


class TestLogicalShiftedRegister:
    def test_and_orr_eor_bic(self, run_a64):
        proc = run_a64('movz x0, #0x0F\n movz x1, #0x33\n and x2, x0, x1\n'
                       ' orr x3, x0, x1\n eor x4, x0, x1\n bic x5, x0, x1')
        assert proc.registers.get_x(2) == 0x03
        assert proc.registers.get_x(3) == 0x3F
        assert proc.registers.get_x(4) == 0x3C
        assert proc.registers.get_x(5) == 0x0C

    def test_mov_register_is_an_orr_alias(self, run_a64):
        proc = run_a64('movz x0, #7\n mov x1, x0')
        assert proc.registers.get_x(1) == 7

    def test_mvn_inverts(self, run_a64):
        proc = run_a64('movz x0, #7\n mvn x1, x0')
        assert proc.registers.get_x(1) == 0xFFFFFFFFFFFFFFF8

    def test_ands_clears_carry_and_overflow(self, run_a64):
        proc = run_a64('movz x0, #0\n movz x1, #0\n ands x2, x0, x1')
        assert proc.registers.pstate.z == 1
        assert proc.registers.pstate.c == 0
        assert proc.registers.pstate.v == 0


class TestAddSubWithCarry:
    def test_carry_propagates(self, run_a64):
        proc = run_a64('movn x0, #0\n movz x1, #1\n adds x2, x0, x1\n adc x3, xzr, xzr')
        assert proc.registers.get_x(2) == 0
        assert proc.registers.get_x(3) == 1


class TestConditionalSelect:
    def test_csel_picks_by_condition(self, run_a64):
        proc = run_a64('movz x0, #5\n movz x1, #3\n cmp x0, x1\n csel x2, x0, x1, gt')
        assert proc.registers.get_x(2) == 5

    def test_cset_materialises_a_flag(self, run_a64):
        proc = run_a64('movz x0, #5\n movz x1, #3\n cmp x0, x1\n cset x2, gt\n cset x3, lt')
        assert proc.registers.get_x(2) == 1
        assert proc.registers.get_x(3) == 0

    def test_csinc_increments_the_else_operand(self, run_a64):
        proc = run_a64('movz x0, #5\n movz x1, #3\n cmp x0, x1\n csinc x2, x0, x1, lt')
        assert proc.registers.get_x(2) == 4


class TestConditionalCompare:
    def test_chained_compare_holds(self, run_a64):
        proc = run_a64('movz x0, #5\n movz x1, #5\n cmp x0, x1\n'
                       ' ccmp x0, #5, #0, eq\n cset x2, eq')
        assert proc.registers.get_x(2) == 1

    def test_flags_substituted_when_condition_fails(self, run_a64):
        # The first compare fails, so CCMP installs its immediate flags instead.
        # The compared immediate is only five bits wide, hence #9 rather than a
        # larger sentinel; it is never evaluated here in any case.
        proc = run_a64('movz x0, #5\n movz x1, #4\n cmp x0, x1\n'
                       ' ccmp x0, #9, #4, eq\n cset x2, eq')
        assert proc.registers.pstate.z == 1
        assert proc.registers.get_x(2) == 1


class TestDataProcessing2Source:
    def test_udiv_and_msub_give_quotient_and_remainder(self, run_a64):
        proc = run_a64('movz x0, #100\n movz x1, #7\n udiv x2, x0, x1\n msub x3, x2, x1, x0')
        assert proc.registers.get_x(2) == 14
        assert proc.registers.get_x(3) == 2

    def test_division_by_zero_yields_zero(self, run_a64):
        # AArch64 removed the divide-by-zero trap; the result is defined as zero.
        proc = run_a64('movz x0, #10\n movz x1, #0\n udiv x2, x0, x1\n sdiv x3, x0, x1')
        assert proc.registers.get_x(2) == 0
        assert proc.registers.get_x(3) == 0

    def test_sdiv_truncates_toward_zero(self, run_a64):
        proc = run_a64('movn x0, #6\n movz x1, #2\n sdiv x2, x0, x1')
        # -7 / 2 truncates to -3, not the -4 that flooring would give.
        assert proc.registers.get_x(2) == 0xFFFFFFFFFFFFFFFD

    def test_variable_shifts(self, run_a64):
        proc = run_a64('movz x0, #1\n movz x1, #4\n lsl x2, x0, x1\n'
                       ' movn x3, #0\n lsr x4, x3, x1\n asr x5, x3, x1')
        assert proc.registers.get_x(2) == 0x10
        assert proc.registers.get_x(4) == 0x0FFFFFFFFFFFFFFF
        assert proc.registers.get_x(5) == 0xFFFFFFFFFFFFFFFF


class TestDataProcessing1Source:
    def test_clz_and_rbit(self, run_a64):
        proc = run_a64('movz x0, #1\n clz x1, x0\n rbit x2, x0')
        assert proc.registers.get_x(1) == 63
        assert proc.registers.get_x(2) == 0x8000000000000000

    def test_rev_reverses_whole_register(self, run_a64):
        proc = run_a64('movz x0, #0x1234\n rev x1, x0')
        assert proc.registers.get_x(1) == 0x3412000000000000

    def test_rev16_reverses_within_halfwords(self, run_a64):
        proc = run_a64('movz x0, #0x1234\n rev16 x1, x0')
        assert proc.registers.get_x(1) == 0x3412

    def test_rev_on_32_bit_register(self, run_a64):
        proc = run_a64('movz w0, #0x1234\n rev w1, w0')
        assert proc.registers.get_x(1) == 0x34120000


class TestDataProcessing3Source:
    def test_mul_and_madd(self, run_a64):
        proc = run_a64('movz x0, #5\n movz x1, #3\n mul x2, x0, x1\n'
                       ' movz x3, #1\n madd x4, x0, x1, x3')
        assert proc.registers.get_x(2) == 15
        assert proc.registers.get_x(4) == 16

    def test_umulh_returns_the_high_half(self, run_a64):
        proc = run_a64('movn x0, #0\n movz x1, #2\n umulh x2, x0, x1')
        # (2**64 - 1) * 2 has 1 in its upper 64 bits.
        assert proc.registers.get_x(2) == 1

    def test_smull_widens_signed_words(self, run_a64):
        proc = run_a64('movn w0, #0\n movz w1, #2\n smull x2, w0, w1')
        # -1 * 2 as a 64-bit signed result.
        assert proc.registers.get_x(2) == 0xFFFFFFFFFFFFFFFE
