from armulator.armv8.bits_ops import add_with_carry, bit_not, shift_reg_value
from armulator.armv8.opcodes.opcode import Opcode


class AddSubShiftedRegister(Opcode):
    """
    ADD/ADDS/SUB/SUBS with a shifted register operand. CMP, CMN and NEG are aliases in
    which the destination or one source is the zero register.
    """

    def __init__(self, instruction, d, n, m, shift_type, shift_amount, setflags,
                 sub_op, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.shift_type = shift_type
        self.shift_amount = shift_amount
        self.setflags = setflags
        self.sub_op = sub_op
        self.datasize = datasize

    def execute(self, processor):
        operand1 = processor.registers.get_x(self.n, self.datasize)
        operand2 = shift_reg_value(
            processor.registers.get_x(self.m, self.datasize),
            self.shift_type, self.shift_amount, self.datasize,
        )
        if self.sub_op:
            operand2 = bit_not(operand2, self.datasize)
            carry_in = 1
        else:
            carry_in = 0
        result, carry, overflow = add_with_carry(operand1, operand2, carry_in, self.datasize)
        if self.setflags:
            processor.registers.set_flags(result, carry, overflow, self.datasize)
        processor.registers.set_x(self.d, result, self.datasize)
