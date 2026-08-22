"""
Data processing - immediate.

Selected by op0 = 100x at instr[28:25], then subdivided by instr[25:23]:

    000, 001    PC-relative addressing
    010         Add/subtract (immediate)
    011         Add/subtract (immediate, with tags) - ARMv8.5, unallocated on Cortex-A57
    100         Logical (immediate)
    101         Move wide (immediate)
    110         Bitfield
    111         Extract
"""

from armulator.armv8.bits_ops import substring
from armulator.armv8.opcodes.concrete.add_sub_immediate import AddSubImmediateA64
from armulator.armv8.opcodes.concrete.bitfield import BitfieldA64
from armulator.armv8.opcodes.concrete.extract import ExtractA64
from armulator.armv8.opcodes.concrete.logical_immediate import LogicalImmediateA64
from armulator.armv8.opcodes.concrete.move_wide_immediate import MoveWideImmediateA64
from armulator.armv8.opcodes.concrete.pc_rel_addressing import PcRelAddressingA64


def decode_instruction(instr):
    op0 = substring(instr, 25, 23)

    if op0 in (0b000, 0b001):
        # PC-relative addressing: ADR, ADRP
        return PcRelAddressingA64
    elif op0 == 0b010:
        # Add/subtract (immediate): ADD, ADDS, SUB, SUBS
        return AddSubImmediateA64
    elif op0 == 0b011:
        # Add/subtract (immediate, with tags) - not implemented on Cortex-A57
        return None
    elif op0 == 0b100:
        # Logical (immediate): AND, ORR, EOR, ANDS
        return LogicalImmediateA64
    elif op0 == 0b101:
        # Move wide (immediate): MOVN, MOVZ, MOVK
        return MoveWideImmediateA64
    elif op0 == 0b110:
        # Bitfield: SBFM, BFM, UBFM
        return BitfieldA64
    else:
        # Extract: EXTR
        return ExtractA64
