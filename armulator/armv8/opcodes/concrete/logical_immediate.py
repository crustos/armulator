from armulator.armv8.bits_ops import bit_at, decode_bit_masks, substring
from armulator.armv8.enums import LogicalOp
from armulator.armv8.opcodes.abstract_opcodes.logical_immediate import LogicalImmediate

_OPS = {0b00: LogicalOp.AND, 0b01: LogicalOp.ORR, 0b10: LogicalOp.EOR, 0b11: LogicalOp.AND}


class LogicalImmediateA64(LogicalImmediate):
    """
    sf opc 1 0 0 1 0 0 N immr imms Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        rd = substring(instr, 4, 0)
        rn = substring(instr, 9, 5)
        imms = substring(instr, 15, 10)
        immr = substring(instr, 21, 16)
        imm_n = bit_at(instr, 22)
        opc = substring(instr, 30, 29)
        datasize = 64 if bit_at(instr, 31) else 32

        # A 64-bit bitmask pattern cannot be encoded in a 32-bit instruction.
        if datasize == 32 and imm_n == 1:
            return None

        masks = decode_bit_masks(datasize, imm_n, imms, immr, True)
        if masks is None:
            return None
        imm, _ = masks

        return LogicalImmediateA64(instr, d=rd, n=rn, imm=imm, op=_OPS[opc],
                                   setflags=(opc == 0b11), datasize=datasize)
