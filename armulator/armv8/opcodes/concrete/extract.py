from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.opcodes.abstract_opcodes.extract import Extract


class ExtractA64(Extract):
    """
    sf op21 1 0 0 1 1 1 N o0 Rm imms Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        rd = substring(instr, 4, 0)
        rn = substring(instr, 9, 5)
        imms = substring(instr, 15, 10)
        rm = substring(instr, 20, 16)
        o0 = bit_at(instr, 21)
        imm_n = bit_at(instr, 22)
        op21 = substring(instr, 30, 29)
        sf = bit_at(instr, 31)
        datasize = 64 if sf else 32

        if op21 != 0b00 or o0 != 0:
            return None
        if imm_n != sf:
            return None
        # The 32-bit form can only extract from within 32 bits.
        if sf == 0 and bit_at(imms, 5):
            return None

        return ExtractA64(instr, d=rd, n=rn, m=rm, lsb=imms, datasize=datasize)
