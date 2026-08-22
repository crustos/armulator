import pytest

from armulator.armv8.bits_ops import (
    asr,
    count_leading_sign_bits,
    count_leading_zeros,
    decode_bit_masks,
    decode_reg_extend,
    highest_set_bit,
    lowest_set_bit,
    replicate,
    reverse_bits,
    ror,
)
from armulator.armv8.enums import ExtendType


class TestDecodeBitMasks:
    @pytest.mark.parametrize('imm_n, imms, immr, expected', [
        (1, 7, 0, 0x00000000000000FF),    # 8 low bits set
        (1, 0, 0, 0x0000000000000001),    # single bit
        (1, 31, 0, 0x00000000FFFFFFFF),   # low 32 bits
        (1, 7, 4, 0xF00000000000000F),    # rotated across the wrap point
        (0, 0b011110, 0, 0x7FFFFFFF7FFFFFFF),  # 32-bit element replicated
        (0, 0b111100, 0, 0x5555555555555555),  # 2-bit element replicated
    ])
    def test_known_patterns(self, imm_n, imms, immr, expected):
        wmask, _ = decode_bit_masks(64, imm_n, imms, immr, True)
        assert wmask == expected

    def test_all_ones_is_reserved_for_immediates(self):
        # An all-ones pattern has no encoding as a logical immediate.
        assert decode_bit_masks(64, 1, 0b111111, 0, True) is None

    def test_32_bit_datasize(self):
        wmask, _ = decode_bit_masks(32, 0, 0, 0, True)
        assert wmask == 0x1


class TestShiftsAndCounts:
    def test_ror_wraps(self):
        assert ror(0xF, 64, 4) == 0xF000000000000000

    def test_ror_by_zero_is_identity(self):
        assert ror(0x1234, 64, 0) == 0x1234

    def test_asr_preserves_sign(self):
        assert asr(0x8000000000000000, 64, 4) == 0xF800000000000000

    def test_count_leading_zeros(self):
        assert count_leading_zeros(1, 64) == 63
        assert count_leading_zeros(0, 64) == 64

    def test_count_leading_sign_bits(self):
        assert count_leading_sign_bits(0xFFFFFFFFFFFFFFF0, 64) == 59
        assert count_leading_sign_bits(0, 64) == 63

    def test_highest_and_lowest_set_bit(self):
        assert highest_set_bit(0x8000000000000000, 64) == 63
        assert highest_set_bit(0, 64) == -1
        assert lowest_set_bit(0x8, 64) == 3
        assert lowest_set_bit(0, 64) == 64

    def test_reverse_bits(self):
        assert reverse_bits(0x8000000000000001, 64) == 0x8000000000000001
        assert reverse_bits(0x1, 64) == 0x8000000000000000

    def test_replicate(self):
        assert replicate(0b01, 2, 8) == 0b01010101


class TestExtendReg:
    def test_sign_extend_word(self):
        assert decode_reg_extend(0xFFFFFFFF, ExtendType.SXTW, 0, 64) == 0xFFFFFFFFFFFFFFFF

    def test_zero_extend_byte_with_shift(self):
        assert decode_reg_extend(0xFF, ExtendType.UXTB, 2, 64) == 0x3FC

    def test_sign_extend_byte(self):
        assert decode_reg_extend(0x80, ExtendType.SXTB, 0, 64) == 0xFFFFFFFFFFFFFF80
