from armulator.armv8.bits_ops import bit_not, shift_reg_value
from armulator.armv8.enums import LogicalOp
from armulator.armv8.opcodes.opcode import Opcode


class LogicalShiftedRegister(Opcode):
    """
    AND/BIC/ORR/ORN/EOR/EON/ANDS/BICS with an optionally shifted second operand.

    MOV (register) is an alias of ORR with XZR as the first operand, and MVN is an alias
    of ORN, so this one implementation carries a great deal of ordinary code.
    """

    def __init__(self, instruction, d, n, m, op, shift_type, shift_amount, invert,
                 setflags, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.op = op
        self.shift_type = shift_type
        self.shift_amount = shift_amount
        self.invert = invert
        self.setflags = setflags
        self.datasize = datasize

    def execute(self, processor):
        operand1 = processor.registers.get_x(self.n, self.datasize)
        operand2 = shift_reg_value(
            processor.registers.get_x(self.m, self.datasize),
            self.shift_type, self.shift_amount, self.datasize,
        )
        if self.invert:
            operand2 = bit_not(operand2, self.datasize)

        if self.op == LogicalOp.AND:
            result = operand1 & operand2
        elif self.op == LogicalOp.ORR:
            result = operand1 | operand2
        else:
            result = operand1 ^ operand2

        if self.setflags:
            # The logical operations always leave C and V clear.
            processor.registers.set_flags(result, 0, 0, self.datasize)
        processor.registers.set_x(self.d, result, self.datasize)
