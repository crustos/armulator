from armulator.armv8.bits_ops import bit_at, bit_not, lower_chunk, ones, ror
from armulator.armv8.opcodes.opcode import Opcode


class Bitfield(Opcode):
    """
    SBFM/BFM/UBFM, and the many aliases built on them (SXTB, UBFX, LSL, LSR, ASR, ...).

    All three share one implementation. The source is rotated right by immr, then two masks
    from DecodeBitMasks decide the outcome: wmask selects which bits of the rotated source
    land in the result, and tmask selects which bits come from that result rather than from
    the extension bits above it. The extension bits are the replicated sign bit for SBFM,
    and the untouched destination for BFM.
    """

    def __init__(self, instruction, d, n, wmask, tmask, imms, immr, extend, inzero, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.wmask = wmask
        self.tmask = tmask
        self.imms = imms
        self.immr = immr
        self.extend = extend
        self.inzero = inzero
        self.datasize = datasize

    def execute(self, processor):
        dst = 0 if self.inzero else processor.registers.get_x(self.d, self.datasize)
        src = processor.registers.get_x(self.n, self.datasize)

        # Bitfield move on the low bits.
        bot = (dst & bit_not(self.wmask, self.datasize)) | (ror(src, self.datasize, self.immr) & self.wmask)

        # Extension bits above the field: the replicated sign bit, or the destination.
        if self.extend:
            top = ones(self.datasize) if bit_at(src, self.imms) else 0
        else:
            top = dst

        result = (top & bit_not(self.tmask, self.datasize)) | (bot & self.tmask)
        processor.registers.set_x(self.d, lower_chunk(result, self.datasize), self.datasize)
