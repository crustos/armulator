from armulator.armv8.bits_ops import bit_at, decode_bit_masks, substring
from armulator.armv8.opcodes.abstract_opcodes.bitfield import Bitfield


class BitfieldA64(Bitfield):
    """
    sf opc 1 0 0 1 1 0 N immr imms Rn Rd

    opc selects SBFM (00), BFM (01) or UBFM (10).
    """

    @staticmethod
    def from_bitarray(instr, processor):
        rd = substring(instr, 4, 0)
        rn = substring(instr, 9, 5)
        imms = substring(instr, 15, 10)
        immr = substring(instr, 21, 16)
        imm_n = bit_at(instr, 22)
        opc = substring(instr, 30, 29)
        sf = bit_at(instr, 31)
        datasize = 64 if sf else 32

        if opc == 0b11:
            return None
        # N must match sf for the bitfield instructions.
        if imm_n != sf:
            return None
        # For the 32-bit forms the immediates must fit in five bits.
        if sf == 0 and (bit_at(immr, 5) or bit_at(imms, 5)):
            return None

        inzero = opc != 0b01   # SBFM and UBFM start from zero, BFM merges
        extend = opc == 0b00   # only SBFM sign extends

        masks = decode_bit_masks(datasize, imm_n, imms, immr, False)
        if masks is None:
            return None
        wmask, tmask = masks

        return BitfieldA64(instr, d=rd, n=rn, wmask=wmask, tmask=tmask, imms=imms,
                           immr=immr, extend=extend, inzero=inzero, datasize=datasize)
