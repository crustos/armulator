from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.enums import MoveWideOp
from armulator.armv8.opcodes.abstract_opcodes.move_wide_immediate import MoveWideImmediate

_OPS = {0b00: MoveWideOp.N, 0b10: MoveWideOp.Z, 0b11: MoveWideOp.K}


class MoveWideImmediateA64(MoveWideImmediate):
    """
    sf opc 1 0 0 1 0 1 hw imm16 Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        rd = substring(instr, 4, 0)
        imm16 = substring(instr, 20, 5)
        hw = substring(instr, 22, 21)
        opc = substring(instr, 30, 29)
        datasize = 64 if bit_at(instr, 31) else 32

        if opc == 0b01:
            return None
        # A 32-bit register only has two 16-bit slices to address.
        if datasize == 32 and bit_at(hw, 1):
            return None

        return MoveWideImmediateA64(instr, d=rd, imm16=imm16, pos=hw << 4,
                                    op=_OPS[opc], datasize=datasize)
