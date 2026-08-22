from armulator.armv8.bits_ops import bit_not, lower_chunk, set_substring
from armulator.armv8.enums import MoveWideOp
from armulator.armv8.opcodes.opcode import Opcode


class MoveWideImmediate(Opcode):
    """
    MOVN/MOVZ/MOVK. MOVN and MOVZ write a whole register, MOVK merges into one 16-bit
    slice and leaves the rest of the register alone.
    """

    def __init__(self, instruction, d, imm16, pos, op, datasize):
        super().__init__(instruction)
        self.d = d
        self.imm16 = imm16
        self.pos = pos
        self.op = op
        self.datasize = datasize

    def execute(self, processor):
        if self.op == MoveWideOp.K:
            result = processor.registers.get_x(self.d, self.datasize)
            result = set_substring(result, self.pos + 15, self.pos, self.imm16)
            result = lower_chunk(result, self.datasize)
        else:
            result = lower_chunk(self.imm16 << self.pos, self.datasize)
            if self.op == MoveWideOp.N:
                result = bit_not(result, self.datasize)
        processor.registers.set_x(self.d, result, self.datasize)
