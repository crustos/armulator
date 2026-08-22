"""
Branches, exception generating and system instructions.

Selected by op0 = 101x at instr[28:25]. The subdivision uses op0 = instr[31:29] together
with op1 = instr[25:22]:

    x00                 Unconditional branch (immediate)
    x01, op1 = 0xxx     Compare and branch (immediate)
    x01, op1 = 1xxx     Test and branch (immediate)
    010, op1 = 0xxx     Conditional branch (immediate)
    110, op1 = 00xx     Exception generation
    110, op1 = 0100     Hints, barriers, PSTATE, system register and system instructions
    110, op1 = 1xxx     Unconditional branch (register)
"""

from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.opcodes.concrete.branches import (
    BranchConditionalA64,
    BranchImmediateA64,
    BranchRegisterA64,
    CompareAndBranchA64,
    ExceptionReturnA64,
    TestAndBranchA64,
)
from armulator.armv8.opcodes.concrete.system import (
    BarrierA64,
    ClrexA64,
    ExceptionGenerationA64,
    HintA64,
    MsrImmediateA64,
    SystemInstructionA64,
    SystemRegisterMoveA64,
)


def decode_instruction(instr):
    op0 = substring(instr, 31, 29)
    op1 = substring(instr, 25, 22)

    if substring(op0, 1, 0) == 0b00:
        # x00 - B and BL
        return BranchImmediateA64
    if substring(op0, 1, 0) == 0b01:
        # x01 - compare and branch, or test and branch
        return TestAndBranchA64 if bit_at(op1, 3) else CompareAndBranchA64
    if op0 == 0b010 and not bit_at(op1, 3):
        # B.cond
        return BranchConditionalA64
    if op0 == 0b110:
        if substring(op1, 3, 2) == 0b00:
            # SVC, HVC, SMC, BRK, HLT
            return ExceptionGenerationA64
        if bit_at(op1, 3):
            # BR, BLR, RET, ERET
            return _decode_branch_register(instr)
        if op1 == 0b0100:
            return _decode_system(instr)
    return None


def _decode_branch_register(instr):
    opc = substring(instr, 24, 21)
    if opc == 0b0100:
        return ExceptionReturnA64
    if opc in (0b0000, 0b0001, 0b0010):
        return BranchRegisterA64
    return None


def _decode_system(instr):
    """
    The system space at instr[31:22] = 1101010100 splits on op0 at instr[20:19], then on
    CRn: 0010 is the hint space, 0011 the barriers, 0100 the writable PSTATE fields.
    """
    system_op0 = substring(instr, 20, 19)
    crn = substring(instr, 15, 12)
    op2 = substring(instr, 7, 5)

    if system_op0 == 0b00:
        if crn == 0b0100:
            # MSR (immediate): DAIFSet, DAIFClr, SPSel
            return MsrImmediateA64
        if crn == 0b0010:
            # NOP, YIELD, WFE, WFI, SEV, SEVL
            return HintA64
        if crn == 0b0011:
            if op2 == 0b010:
                return ClrexA64
            return BarrierA64
        return None
    if system_op0 == 0b01:
        # SYS and SYSL: cache, TLB and address translation maintenance.
        return SystemInstructionA64
    # 1x - MRS and MSR (register)
    return SystemRegisterMoveA64
