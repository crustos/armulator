"""
Data processing - register.

Selected by op0 = x101 at instr[28:25]. Within the group, instr[28] separates the two
halves and op2 = instr[24:21] picks the class:

  instr[28] = 0
      op2 = 0xxx      Logical (shifted register)
      op2 = 1xx0      Add/subtract (shifted register)
      op2 = 1xx1      Add/subtract (extended register)

  instr[28] = 1
      op2 = 0000      Add/subtract (with carry)
      op2 = 0010      Conditional compare (register or immediate)
      op2 = 0100      Conditional select
      op2 = 0110      Data processing (1 or 2 source, on instr[30])
      op2 = 1xxx      Data processing (3 source)
"""

from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.opcodes.concrete.data_processing_register import (
    AddSubExtendedRegisterA64,
    AddSubShiftedRegisterA64,
    AddSubWithCarryA64,
    ConditionalCompareA64,
    ConditionalSelectA64,
    DataProcessing1SourceA64,
    DataProcessing2SourceA64,
    DataProcessing3SourceA64,
    LogicalShiftedRegisterA64,
)


def decode_instruction(instr):
    op2 = substring(instr, 24, 21)

    if not bit_at(instr, 28):
        if not bit_at(op2, 3):
            return LogicalShiftedRegisterA64
        if bit_at(op2, 0):
            return AddSubExtendedRegisterA64
        return AddSubShiftedRegisterA64

    if op2 == 0b0000:
        return AddSubWithCarryA64
    if op2 == 0b0010:
        return ConditionalCompareA64
    if op2 == 0b0100:
        return ConditionalSelectA64
    if op2 == 0b0110:
        # instr[30] distinguishes the one and two source forms.
        return DataProcessing1SourceA64 if bit_at(instr, 30) else DataProcessing2SourceA64
    if bit_at(op2, 3):
        return DataProcessing3SourceA64
    return None
