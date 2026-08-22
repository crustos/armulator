from armulator.armv8.bits_ops import bit_at, chain, sign_extend, substring
from armulator.armv8.opcodes.abstract_opcodes.pc_rel_addressing import PcRelAddressing


class PcRelAddressingA64(PcRelAddressing):
    """
    op immlo 1 0 0 0 0 immhi Rd

    The 21-bit offset is split across the encoding: immhi holds the top 19 bits and immlo
    the bottom 2. For ADRP the offset is scaled by the 4KB page size.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        rd = substring(instr, 4, 0)
        immhi = substring(instr, 23, 5)
        immlo = substring(instr, 30, 29)
        page = bit_at(instr, 31)

        imm21 = chain(immhi, immlo, 2)
        if page:
            imm = sign_extend(imm21 << 12, 33, 64)
        else:
            imm = sign_extend(imm21, 21, 64)

        return PcRelAddressingA64(instr, d=rd, imm=imm, page=page)
