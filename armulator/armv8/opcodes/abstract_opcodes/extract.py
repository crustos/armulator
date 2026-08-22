from armulator.armv8.bits_ops import chain, lower_chunk, lsr
from armulator.armv8.opcodes.opcode import Opcode


class Extract(Opcode):
    """
    EXTR - concatenate two registers and extract a register-width field starting at lsb.
    ROR (immediate) is the alias where both source registers are the same.
    """

    def __init__(self, instruction, d, n, m, lsb, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.lsb = lsb
        self.datasize = datasize

    def execute(self, processor):
        operand1 = processor.registers.get_x(self.n, self.datasize)
        operand2 = processor.registers.get_x(self.m, self.datasize)
        concat = chain(operand1, operand2, self.datasize)
        result = lower_chunk(lsr(concat, self.datasize * 2, self.lsb), self.datasize)
        processor.registers.set_x(self.d, result, self.datasize)
