"""
SIMD and floating point instructions.

Every snippet here needs ``fp=True`` on the runner: CPACR_EL1 resets to trapping SIMD
and floating point, so the register file is unreachable until it is enabled.
"""

from armulator.armv8 import fp_ops
from armulator.armv8.enums import EL

BUF = 0x8000


def as_f32(bits):
    return fp_ops.bits_to_float(bits, 32)


def as_f64(bits):
    return fp_ops.bits_to_float(bits, 64)


def lanes(value, element_size, count):
    mask = (1 << element_size) - 1
    return [(value >> (i * element_size)) & mask for i in range(count)]


class TestAccessTrap:
    def test_fp_traps_before_cpacr_is_enabled(self, run_a64):
        # Firmware must enable FP explicitly, exactly as on real hardware.
        proc, _ = run_a64.build('fmov s0, #1.0')
        proc.registers.vbar[EL.EL1] = 0x200
        proc.emulate_cycle()
        assert proc.registers.esr[EL.EL1] >> 26 == 0x07
        assert proc.registers.get_pc() == 0x400

    def test_fp_works_once_enabled(self, run_a64):
        proc = run_a64('fmov s0, #1.0', fp=True)
        assert as_f32(proc.registers.get_v(0, 32)) == 1.0


class TestScalarArithmetic:
    def test_single_precision(self, run_a64):
        proc = run_a64('fmov s0, #1.5\n fmov s1, #2.5\n fadd s2, s0, s1\n'
                       ' fsub s3, s1, s0\n fmul s4, s0, s1\n fdiv s5, s1, s0', fp=True)
        assert as_f32(proc.registers.get_v(2, 32)) == 4.0
        assert as_f32(proc.registers.get_v(3, 32)) == 1.0
        assert as_f32(proc.registers.get_v(4, 32)) == 3.75

    def test_double_precision(self, run_a64):
        proc = run_a64('fmov d0, #1.0\n fmov d1, #2.0\n fdiv d2, d0, d1\n fsqrt d3, d1',
                       fp=True)
        assert as_f64(proc.registers.get_v(2, 64)) == 0.5
        assert round(as_f64(proc.registers.get_v(3, 64)), 6) == 1.414214

    def test_abs_and_negate(self, run_a64):
        proc = run_a64('fmov d0, #1.0\n fneg d1, d0\n fabs d2, d1', fp=True)
        assert as_f64(proc.registers.get_v(1, 64)) == -1.0
        assert as_f64(proc.registers.get_v(2, 64)) == 1.0

    def test_fused_multiply_add(self, run_a64):
        proc = run_a64('fmov s0, #1.0\n fmov s1, #2.0\n fmadd s2, s0, s1, s1\n'
                       ' fmsub s3, s0, s1, s1', fp=True)
        assert as_f32(proc.registers.get_v(2, 32)) == 4.0     # 2 + 1*2
        assert as_f32(proc.registers.get_v(3, 32)) == 0.0     # 2 - 1*2


class TestScalarCompareAndSelect:
    def test_compare_sets_condition_flags(self, run_a64):
        proc = run_a64('fmov s0, #1.0\n fmov s1, #2.0\n fcmp s0, s1\n cset w0, mi\n'
                       ' fcmp s1, s0\n cset w1, gt\n fcmp s0, s0\n cset w2, eq', fp=True)
        assert proc.registers.get_x(0) == 1
        assert proc.registers.get_x(1) == 1
        assert proc.registers.get_x(2) == 1

    def test_compare_with_zero(self, run_a64):
        proc = run_a64('fmov d0, #1.0\n fcmp d0, #0.0\n cset w0, gt', fp=True)
        assert proc.registers.get_x(0) == 1

    def test_fcsel(self, run_a64):
        proc = run_a64('fmov s0, #1.0\n fmov s1, #2.0\n fcmp s0, s1\n'
                       ' fcsel s2, s0, s1, mi', fp=True)
        assert as_f32(proc.registers.get_v(2, 32)) == 1.0


class TestConversion:
    def test_integer_to_float_and_back(self, run_a64):
        proc = run_a64('movz x0, #42\n scvtf d0, x0\n fcvtzs x1, d0', fp=True)
        assert as_f64(proc.registers.get_v(0, 64)) == 42.0
        assert proc.registers.get_x(1) == 42

    def test_precision_conversion(self, run_a64):
        proc = run_a64('fmov s0, #1.5\n fcvt d1, s0\n fcvt s2, d1', fp=True)
        assert as_f64(proc.registers.get_v(1, 64)) == 1.5
        assert as_f32(proc.registers.get_v(2, 32)) == 1.5

    def test_fmov_reinterprets_rather_than_converting(self, run_a64):
        # FMOV moves bits; SCVTF converts numerically. Confusing them is a classic bug.
        proc = run_a64('fmov d0, #1.0\n fmov x0, d0\n movz x1, #1\n scvtf d1, x1\n'
                       ' fmov d2, x1', fp=True)
        assert proc.registers.get_x(0) == 0x3FF0000000000000
        assert as_f64(proc.registers.get_v(1, 64)) == 1.0
        assert proc.registers.get_v(2, 64) == 1            # a denormal, not 1.0

    def test_fmov_to_upper_half_of_vector(self, run_a64):
        proc = run_a64('movz x0, #5\n movi v0.4s, #0\n fmov v0.d[1], x0\n'
                       ' fmov x1, v0.d[1]', fp=True)
        assert proc.registers.get_x(1) == 5


class TestFpLoadStore:
    def test_q_register_round_trip(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movi v0.4s, #1\n str q0, [x1]\n'
                       ' ldr q1, [x1]\n ldr x2, [x1]\n ldr x3, [x1, #8]', fp=True)
        assert proc.registers.get_v(1) == 0x00000001000000010000000100000001
        assert proc.registers.get_x(2) == 0x0000000100000001
        assert proc.registers.get_x(3) == 0x0000000100000001

    def test_d_and_s_accesses(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n fmov d0, #1.0\n str d0, [x1]\n'
                       ' ldr d1, [x1]\n ldr s2, [x1]', fp=True)
        assert as_f64(proc.registers.get_v(1, 64)) == 1.0
        assert proc.registers.get_v(2, 32) == 0

    def test_narrow_write_zeroes_the_rest_of_the_register(self, run_a64):
        # Writing D0 must clear bits 127:64 of V0, mirroring the 32-bit rule for X.
        proc = run_a64('movi v0.16b, #255\n fmov d1, #1.0\n fmov d0, d1', fp=True)
        assert proc.registers.get_v(0) == 0x3FF0000000000000

    def test_pair_of_q_registers(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movi v0.4s, #7\n movi v1.4s, #9\n'
                       ' stp q0, q1, [x1]\n ldp q2, q3, [x1]', fp=True)
        assert proc.registers.get_v(2) == 0x00000007000000070000000700000007
        assert proc.registers.get_v(3) == 0x00000009000000090000000900000009

    def test_post_indexed_q_load(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movi v0.4s, #3\n str q0, [x1]\n'
                       ' ldr q1, [x1], #16', fp=True)
        assert proc.registers.get_x(1) == BUF + 16
        assert proc.registers.get_v(1) == 0x00000003000000030000000300000003


class TestVectorArithmetic:
    def test_integer_lanes(self, run_a64):
        proc = run_a64('movi v0.4s, #3\n movi v1.4s, #4\n add v2.4s, v0.4s, v1.4s\n'
                       ' sub v3.4s, v1.4s, v0.4s\n mul v4.4s, v0.4s, v1.4s', fp=True)
        assert lanes(proc.registers.get_v(2), 32, 4) == [7, 7, 7, 7]
        assert lanes(proc.registers.get_v(3), 32, 4) == [1, 1, 1, 1]
        assert lanes(proc.registers.get_v(4), 32, 4) == [12, 12, 12, 12]

    def test_64_bit_form_zeroes_the_upper_half(self, run_a64):
        proc = run_a64('movi v0.16b, #255\n movi v1.8b, #1\n movi v2.8b, #1\n'
                       ' add v0.8b, v1.8b, v2.8b', fp=True)
        assert proc.registers.get_v(0) >> 64 == 0

    def test_float_lanes(self, run_a64):
        proc = run_a64('fmov v0.4s, #1.5\n fmov v1.4s, #2.5\n fadd v2.4s, v0.4s, v1.4s\n'
                       ' fmul v3.4s, v0.4s, v1.4s', fp=True)
        assert [as_f32(x) for x in lanes(proc.registers.get_v(2), 32, 4)] == [4.0] * 4
        assert [as_f32(x) for x in lanes(proc.registers.get_v(3), 32, 4)] == [3.75] * 4

    def test_double_precision_lanes(self, run_a64):
        proc = run_a64('fmov v0.2d, #1.0\n fmov v1.2d, #2.0\n fadd v2.2d, v0.2d, v1.2d',
                       fp=True)
        assert [as_f64(x) for x in lanes(proc.registers.get_v(2), 64, 2)] == [3.0, 3.0]

    def test_comparisons_produce_lane_masks(self, run_a64):
        proc = run_a64('movi v0.4s, #3\n movi v1.4s, #3\n cmeq v2.4s, v0.4s, v1.4s\n'
                       ' movi v3.4s, #4\n cmeq v4.4s, v0.4s, v3.4s', fp=True)
        assert lanes(proc.registers.get_v(2), 32, 4) == [0xFFFFFFFF] * 4
        assert lanes(proc.registers.get_v(4), 32, 4) == [0] * 4

    def test_bitwise_operations(self, run_a64):
        proc = run_a64('movi v0.16b, #255\n movi v1.4s, #0\n and v2.16b, v0.16b, v1.16b\n'
                       ' orr v3.16b, v0.16b, v1.16b\n eor v4.16b, v0.16b, v0.16b', fp=True)
        assert proc.registers.get_v(2) == 0
        assert proc.registers.get_v(3) == (1 << 128) - 1
        assert proc.registers.get_v(4) == 0


class TestVectorMoves:
    def test_duplicate_from_general_register(self, run_a64):
        proc = run_a64('movz x0, #0xAB\n dup v0.16b, w0', fp=True)
        assert proc.registers.get_v(0) == int('AB' * 16, 16)

    def test_duplicate_from_lane(self, run_a64):
        proc = run_a64('movz x0, #7\n dup v0.4s, w0\n dup v1.4s, v0.s[2]', fp=True)
        assert lanes(proc.registers.get_v(1), 32, 4) == [7, 7, 7, 7]

    def test_insert_preserves_other_lanes(self, run_a64):
        # INS is the one vector write that must not zero the rest of the register.
        proc = run_a64('movi v0.16b, #255\n movz x1, #0\n ins v0.d[1], x1', fp=True)
        assert proc.registers.get_v(0) == 0x0000000000000000FFFFFFFFFFFFFFFF

    def test_unsigned_move_out_of_a_lane(self, run_a64):
        proc = run_a64('movz x0, #0xAB\n dup v0.16b, w0\n umov w1, v0.s[0]', fp=True)
        assert proc.registers.get_x(1) == 0xABABABAB

    def test_signed_move_sign_extends(self, run_a64):
        proc = run_a64('movz x0, #0x80\n dup v0.16b, w0\n smov x1, v0.b[3]', fp=True)
        assert proc.registers.get_x(1) == 0xFFFFFFFFFFFFFF80

    def test_movi_and_mvni(self, run_a64):
        proc = run_a64('movi v0.4s, #1\n mvni v1.4s, #1', fp=True)
        assert lanes(proc.registers.get_v(0), 32, 4) == [1] * 4
        assert lanes(proc.registers.get_v(1), 32, 4) == [0xFFFFFFFE] * 4


class TestCompilerStylePatterns:
    def test_q_register_memcpy_loop(self, run_a64):
        # The shape an optimised memcpy takes: 16 bytes per iteration through Q0.
        def seed(proc):
            for offset in range(64):
                proc.mem_set(BUF + offset, 1, offset)

        proc = run_a64(f'''
                movz x0, #{BUF + 0x1000}
                movz x1, #{BUF}
                movz w2, #4
        copy:   ldr  q0, [x1], #16
                str  q0, [x0], #16
                subs w2, w2, #1
                b.ne copy
        ''', steps=40, fp=True, setup=seed)

        copied = [proc.mem_get(BUF + 0x1000 + offset, 1) for offset in range(64)]
        assert copied == list(range(64))

    def test_float_accumulation_loop(self, run_a64):
        proc = run_a64('''
                fmov d0, #0.0
                fmov d1, #0.5
                movz w2, #4
        acc:    fadd d0, d0, d1
                subs w2, w2, #1
                b.ne acc
        ''', steps=30, fp=True)
        assert as_f64(proc.registers.get_v(0, 64)) == 2.0


class TestStructureLoadStore:
    def test_ld1_st1_move_two_registers(self, run_a64):
        # GCC emits this pair for structure assignment.
        proc = run_a64(f'movz x1, #{BUF}\n movi v0.4s, #1\n movi v1.4s, #2\n'
                       f' st1 {{v0.16b, v1.16b}}, [x1]\n'
                       f' ld1 {{v2.16b, v3.16b}}, [x1]', fp=True)
        assert lanes(proc.registers.get_v(2), 32, 4) == [1] * 4
        assert lanes(proc.registers.get_v(3), 32, 4) == [2] * 4

    def test_ld1_single_register(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movi v0.4s, #5\n st1 {{v0.16b}}, [x1]\n'
                       f' ld1 {{v1.16b}}, [x1]', fp=True)
        assert lanes(proc.registers.get_v(1), 32, 4) == [5] * 4

    def test_ld1_post_index_advances_by_the_transfer_size(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movi v0.4s, #1\n'
                       f' st1 {{v0.16b, v1.16b}}, [x1], #32', fp=True)
        assert proc.registers.get_x(1) == BUF + 32


class TestShiftByImmediate:
    def test_scalar_shift_right(self, run_a64):
        # The form GCC uses to extract the upper half of a packed pair.
        proc = run_a64('movz x0, #1, lsl #48\n fmov d0, x0\n ushr d1, d0, #32', fp=True)
        assert proc.registers.get_v(1, 64) == 0x10000

    def test_vector_shift_left(self, run_a64):
        proc = run_a64('movi v0.4s, #1\n shl v1.4s, v0.4s, #4', fp=True)
        assert lanes(proc.registers.get_v(1), 32, 4) == [0x10] * 4

    def test_arithmetic_shift_preserves_sign(self, run_a64):
        proc = run_a64('mvni v0.4s, #0\n sshr v1.4s, v0.4s, #4\n ushr v2.4s, v0.4s, #4',
                       fp=True)
        assert lanes(proc.registers.get_v(1), 32, 4) == [0xFFFFFFFF] * 4
        assert lanes(proc.registers.get_v(2), 32, 4) == [0x0FFFFFFF] * 4


class TestByElement:
    def test_multiply_by_a_broadcast_lane(self, run_a64):
        proc = run_a64('fmov v0.2d, #2.0\n fmov v4.2d, #2.5\n'
                       ' fmul v0.2d, v0.2d, v4.d[0]', fp=True)
        assert [as_f64(x) for x in lanes(proc.registers.get_v(0), 64, 2)] == [5.0, 5.0]

    def test_multiply_accumulate_by_element(self, run_a64):
        proc = run_a64('fmov v0.4s, #1.0\n fmov v1.4s, #2.0\n fmov v2.4s, #3.0\n'
                       ' fmla v0.4s, v1.4s, v2.s[0]', fp=True)
        # Each lane becomes 1.0 + 2.0 * 3.0
        assert [as_f32(x) for x in lanes(proc.registers.get_v(0), 32, 4)] == [7.0] * 4


class TestAdvancedSimdScalarIsNotMisdecoded:
    def test_scalar_simd_does_not_execute_as_floating_point(self, run_a64):
        # USHR sits in the Advanced SIMD scalar space, which overlaps the scalar
        # floating point encodings except for bit 30. Decoding it as an FMADD would
        # execute silently and produce wrong answers rather than faulting.
        from armulator.armv8.opcodes.decode_instruction import decode_instruction
        proc, _ = run_a64.build('ushr d0, d0, #32')
        opcode_class = decode_instruction(0x7F600400, proc)
        assert opcode_class is not None
        assert 'Fp' not in opcode_class.__name__
        assert 'Shift' in opcode_class.__name__


class TestStructureDeinterleaving:
    def _seed(self, values, size=4, base=BUF):
        def setup(proc):
            for index, value in enumerate(values):
                proc.mem_set(base + index * size, size, value)
        return setup

    def test_ld2_splits_pairs(self, run_a64):
        # Memory holds an interleaved array of pairs; LD2 turns it into two vectors.
        proc = run_a64(f'movz x1, #{BUF}\n ld2 {{v0.4s, v1.4s}}, [x1]',
                       fp=True, setup=self._seed(range(8)))
        assert lanes(proc.registers.get_v(0), 32, 4) == [0, 2, 4, 6]
        assert lanes(proc.registers.get_v(1), 32, 4) == [1, 3, 5, 7]

    def test_ld3_splits_three_member_structures(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n ld3 {{v0.8b, v1.8b, v2.8b}}, [x1]',
                       fp=True, setup=self._seed(range(24), size=1))
        assert lanes(proc.registers.get_v(0, 64), 8, 8) == [0, 3, 6, 9, 12, 15, 18, 21]
        assert lanes(proc.registers.get_v(1, 64), 8, 8) == [1, 4, 7, 10, 13, 16, 19, 22]
        assert lanes(proc.registers.get_v(2, 64), 8, 8) == [2, 5, 8, 11, 14, 17, 20, 23]

    def test_a_64_bit_form_zeroes_the_upper_half(self, run_a64):
        proc = run_a64(f'movi v0.16b, #255\n movz x1, #{BUF}\n'
                       f' ld3 {{v0.8b, v1.8b, v2.8b}}, [x1]',
                       fp=True, setup=self._seed(range(24), size=1))
        assert proc.registers.get_v(0) >> 64 == 0

    def test_st4_restores_the_interleaving(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movz x2, #{BUF + 0x100}\n'
                       f' ld4 {{v0.4h, v1.4h, v2.4h, v3.4h}}, [x1]\n'
                       f' st4 {{v0.4h, v1.4h, v2.4h, v3.4h}}, [x2]',
                       fp=True, setup=self._seed(range(16), size=2))
        restored = [proc.mem_get(BUF + 0x100 + index * 2, 2) for index in range(16)]
        assert restored == list(range(16))

    def test_ld1_is_still_contiguous(self, run_a64):
        # LD1 shares the encoding space but does not de-interleave.
        proc = run_a64(f'movz x1, #{BUF}\n ld1 {{v0.4s, v1.4s}}, [x1]',
                       fp=True, setup=self._seed(range(8)))
        assert lanes(proc.registers.get_v(0), 32, 4) == [0, 1, 2, 3]
        assert lanes(proc.registers.get_v(1), 32, 4) == [4, 5, 6, 7]

    def test_post_index_advances_by_the_whole_transfer(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n ld3 {{v0.16b, v1.16b, v2.16b}}, [x1], #48',
                       fp=True)
        assert proc.registers.get_x(1) == BUF + 48


class TestSaturatingArithmetic:
    def test_signed_add_saturates_instead_of_wrapping(self, run_a64):
        proc = run_a64('movi v0.16b, #0x7f\n movi v1.16b, #0x10\n'
                       ' add v2.16b, v0.16b, v1.16b\n sqadd v3.16b, v0.16b, v1.16b',
                       fp=True)
        assert lanes(proc.registers.get_v(2), 8, 16)[0] == 0x8F      # wraps to negative
        assert lanes(proc.registers.get_v(3), 8, 16)[0] == 0x7F      # pins at the limit

    def test_unsigned_add_saturates(self, run_a64):
        proc = run_a64('movi v0.16b, #0xff\n movi v1.16b, #0x10\n'
                       ' uqadd v2.16b, v0.16b, v1.16b', fp=True)
        assert lanes(proc.registers.get_v(2), 8, 16)[0] == 0xFF

    def test_unsigned_subtract_saturates_at_zero(self, run_a64):
        proc = run_a64('movi v0.16b, #0xff\n movi v1.16b, #0x10\n'
                       ' uqsub v2.16b, v1.16b, v0.16b', fp=True)
        assert lanes(proc.registers.get_v(2), 8, 16)[0] == 0

    def test_signed_subtract_saturates_at_the_negative_limit(self, run_a64):
        proc = run_a64('movi v0.16b, #0x80\n movi v1.16b, #0x10\n'
                       ' sqsub v2.16b, v0.16b, v1.16b', fp=True)
        assert lanes(proc.registers.get_v(2), 8, 16)[0] == 0x80


class TestAcrossLaneReductions:
    def _vector(self, values):
        """Build a snippet loading four word lanes."""
        moves = '\n'.join(
            f' movz w{index}, #{value}\n mov v0.s[{index}], w{index}'
            for index, value in enumerate(values))
        return moves

    def test_addv_sums_every_lane(self, run_a64):
        proc = run_a64(self._vector([1, 2, 5, 9]) + '\n addv s1, v0.4s', fp=True)
        assert proc.registers.get_v(1, 32) == 17

    def test_the_result_is_a_scalar(self, run_a64):
        # Everything above the destination's lowest lane is cleared.
        proc = run_a64(self._vector([1, 2, 5, 9]) + '\n addv s1, v0.4s', fp=True)
        assert proc.registers.get_v(1) >> 32 == 0

    def test_unsigned_max_and_min(self, run_a64):
        proc = run_a64(self._vector([3, 17, 5, 9]) +
                       '\n umaxv s1, v0.4s\n uminv s2, v0.4s', fp=True)
        assert proc.registers.get_v(1, 32) == 17
        assert proc.registers.get_v(2, 32) == 3

    def test_signed_and_unsigned_reductions_differ(self, run_a64):
        # Every lane is 0xFF except one, which is 1. Signed, 0xFF is -1 and the
        # maximum is 1; unsigned, 0xFF is the maximum.
        source = ('movn w0, #0\n dup v0.8b, w0\n movz w1, #1\n mov v0.b[3], w1\n'
                  ' smaxv b1, v0.8b\n sminv b2, v0.8b\n umaxv b3, v0.8b')
        proc = run_a64(source, fp=True)
        assert proc.registers.get_v(1, 8) == 1
        assert proc.registers.get_v(2, 8) == 0xFF
        assert proc.registers.get_v(3, 8) == 0xFF


class TestPairwiseAdd:
    def test_addp_folds_adjacent_lanes(self, run_a64):
        source = '\n'.join(f' movz w{i}, #{i + 1}\n mov v0.s[{i}], w{i}' for i in range(4))
        proc = run_a64(source + '\n addp v2.4s, v0.4s, v0.4s', fp=True)
        # Low half from the first source, high half from the second.
        assert lanes(proc.registers.get_v(2), 32, 4) == [3, 7, 3, 7]
