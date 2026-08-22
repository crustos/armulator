"""
Concrete encodings for the branch instructions.
"""

from armulator.armv8.bits_ops import bit_at, chain, sign_extend, substring
from armulator.armv8.opcodes.abstract_opcodes.branch_conditional import BranchConditional
from armulator.armv8.opcodes.abstract_opcodes.branch_immediate import BranchImmediate
from armulator.armv8.opcodes.abstract_opcodes.branch_register import BranchRegister
from armulator.armv8.opcodes.abstract_opcodes.compare_and_branch import CompareAndBranch
from armulator.armv8.opcodes.abstract_opcodes.exception_return import ExceptionReturn
from armulator.armv8.opcodes.abstract_opcodes.test_and_branch import TestAndBranch


class BranchImmediateA64(BranchImmediate):
    """
    op 0 0 1 0 1 imm26 - B when op is 0, BL when it is 1.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        imm26 = substring(instr, 25, 0)
        # The offset is a word count, so it is scaled by four before sign extension.
        offset = sign_extend(imm26 << 2, 28, 64)
        return BranchImmediateA64(instr, offset=offset, with_link=bit_at(instr, 31))


class BranchConditionalA64(BranchConditional):
    """
    0 1 0 1 0 1 0 0 imm19 o0 cond
    """

    @staticmethod
    def from_bitarray(instr, processor):
        if bit_at(instr, 4):
            return None
        imm19 = substring(instr, 23, 5)
        offset = sign_extend(imm19 << 2, 21, 64)
        return BranchConditionalA64(instr, offset=offset, condition=substring(instr, 3, 0))


class CompareAndBranchA64(CompareAndBranch):
    """
    sf 0 1 1 0 1 0 op imm19 Rt
    """

    @staticmethod
    def from_bitarray(instr, processor):
        imm19 = substring(instr, 23, 5)
        offset = sign_extend(imm19 << 2, 21, 64)
        return CompareAndBranchA64(
            instr,
            t=substring(instr, 4, 0),
            offset=offset,
            branch_if_nonzero=bit_at(instr, 24),
            datasize=64 if bit_at(instr, 31) else 32,
        )


class TestAndBranchA64(TestAndBranch):
    """
    b5 0 1 1 0 1 1 op b40 imm14 Rt

    The bit number is b5:b40, so the top bit doubles as the operand size selector.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        b5 = bit_at(instr, 31)
        b40 = substring(instr, 23, 19)
        imm14 = substring(instr, 18, 5)
        offset = sign_extend(imm14 << 2, 16, 64)
        return TestAndBranchA64(
            instr,
            t=substring(instr, 4, 0),
            bit_pos=chain(b5, b40, 5),
            offset=offset,
            branch_if_set=bit_at(instr, 24),
            datasize=64 if b5 else 32,
        )


class BranchRegisterA64(BranchRegister):
    """
    1 1 0 1 0 1 1 0 opc op2 op3 Rn op4 - BR, BLR and RET.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        opc = substring(instr, 24, 21)
        if substring(instr, 20, 16) != 0b11111 or substring(instr, 15, 10) != 0:
            return None
        if substring(instr, 4, 0) != 0:
            return None
        if opc not in (0b0000, 0b0001, 0b0010):
            return None
        return BranchRegisterA64(
            instr, n=substring(instr, 9, 5), with_link=(opc == 0b0001)
        )


class ExceptionReturnA64(ExceptionReturn):
    """
    1 1 0 1 0 1 1 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 - ERET.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        if substring(instr, 20, 16) != 0b11111 or substring(instr, 15, 10) != 0:
            return None
        if substring(instr, 9, 5) != 0b11111 or substring(instr, 4, 0) != 0:
            return None
        return ExceptionReturnA64(instr)
