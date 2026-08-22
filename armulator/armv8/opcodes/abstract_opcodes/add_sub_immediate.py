from armulator.armv8.bits_ops import add_with_carry, bit_not
from armulator.armv8.opcodes.opcode import Opcode


class AddSubImmediate(Opcode):
    """
    ADD/ADDS/SUB/SUBS (immediate), and the CMP/CMN/MOV aliases built on them.

    Subtraction is performed as an addition of the inverted operand with a carry in, which
    is what makes SUBS set the carry flag as "not borrow".
    """

    def __init__(self, instruction, d, n, imm, setflags, sub_op, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.imm = imm
        self.setflags = setflags
        self.sub_op = sub_op
        self.datasize = datasize

    def execute(self, processor):
        operand1 = processor.registers.get_reg_or_sp(self.n, self.datasize)
        operand2 = self.imm
        if self.sub_op:
            operand2 = bit_not(operand2, self.datasize)
            carry_in = 1
        else:
            carry_in = 0
        result, carry, overflow = add_with_carry(operand1, operand2, carry_in, self.datasize)
        if self.setflags:
            processor.registers.set_flags(result, carry, overflow, self.datasize)
            # When the flags are set, register 31 is the zero register rather than SP.
            processor.registers.set_x(self.d, result, self.datasize)
        else:
            processor.registers.set_reg_or_sp(self.d, result, self.datasize)
