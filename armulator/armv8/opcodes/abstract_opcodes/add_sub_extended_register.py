from armulator.armv8.bits_ops import add_with_carry, bit_not, decode_reg_extend
from armulator.armv8.opcodes.opcode import Opcode


class AddSubExtendedRegister(Opcode):
    """
    ADD/ADDS/SUB/SUBS where the second operand is extended from a narrower width before
    use. This is the only add/subtract form that can take SP as an operand, which is why
    a compiler reaches for it when doing stack arithmetic with a 32-bit index.
    """

    def __init__(self, instruction, d, n, m, extend_type, shift, setflags, sub_op, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.extend_type = extend_type
        self.shift = shift
        self.setflags = setflags
        self.sub_op = sub_op
        self.datasize = datasize

    def execute(self, processor):
        operand1 = processor.registers.get_reg_or_sp(self.n, self.datasize)
        operand2 = decode_reg_extend(
            processor.registers.get_x(self.m, self.datasize),
            self.extend_type, self.shift, self.datasize,
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
        else:
            processor.registers.set_reg_or_sp(self.d, result, self.datasize)
