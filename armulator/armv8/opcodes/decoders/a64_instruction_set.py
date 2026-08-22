"""
Top level A64 encoding.

Every A64 instruction is 32 bits and the main class is selected by a single 4-bit field,
op0 at instr[28:25]. This is far more regular than the AArch32 tree, so the whole top
level fits in one dispatch.

    op0     group
    0000    Reserved
    0001    Unallocated
    0010    Unallocated on Cortex-A57 (SVE on later cores)
    0011    Unallocated
    100x    Data processing - immediate
    101x    Branches, exception generating and system instructions
    x1x0    Loads and stores
    x101    Data processing - register
    x111    Data processing - SIMD and floating point
"""

from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.opcodes.decoders import (
    branches_exception_system,
    data_processing_immediate,
    data_processing_register,
    data_processing_simd_fp,
    loads_and_stores,
)


def decode_instruction(instr, processor=None):
    op0 = substring(instr, 28, 25)

    if substring(op0, 3, 1) == 0b100:
        # 100x - Data processing - immediate
        return data_processing_immediate.decode_instruction(instr)
    elif substring(op0, 3, 1) == 0b101:
        # 101x - Branches, exception generating and system instructions
        return branches_exception_system.decode_instruction(instr)
    elif bit_at(op0, 2) and not bit_at(op0, 0):
        # x1x0 - Loads and stores
        return loads_and_stores.decode_instruction(instr)
    elif bit_at(op0, 2) and substring(op0, 1, 0) == 0b01:
        # x101 - Data processing - register
        return data_processing_register.decode_instruction(instr)
    elif bit_at(op0, 2) and substring(op0, 1, 0) == 0b11:
        # x111 - Data processing - SIMD and floating point
        return data_processing_simd_fp.decode_instruction(instr)
    else:
        # 00xx - Reserved and unallocated
        return None
