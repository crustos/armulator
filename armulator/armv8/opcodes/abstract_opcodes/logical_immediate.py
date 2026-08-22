from armulator.armv8.enums import LogicalOp
from armulator.armv8.opcodes.opcode import Opcode


class LogicalImmediate(Opcode):
    """
    AND/ORR/EOR/ANDS (immediate). The immediate has already been expanded from the
    N:immr:imms bitmask encoding by the decoder.
    """

    def __init__(self, instruction, d, n, imm, op, setflags, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.imm = imm
        self.op = op
        self.setflags = setflags
        self.datasize = datasize

    def execute(self, processor):
        operand1 = processor.registers.get_x(self.n, self.datasize)
        operand2 = self.imm
        if self.op == LogicalOp.AND:
            result = operand1 & operand2
        elif self.op == LogicalOp.ORR:
            result = operand1 | operand2
        else:
            result = operand1 ^ operand2
        if self.setflags:
            # The logical instructions always clear C and V.
            processor.registers.set_flags(result, 0, 0, self.datasize)
            processor.registers.set_x(self.d, result, self.datasize)
        else:
            processor.registers.set_reg_or_sp(self.d, result, self.datasize)
