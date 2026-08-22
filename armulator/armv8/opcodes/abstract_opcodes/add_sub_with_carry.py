from armulator.armv8.bits_ops import add_with_carry, bit_not
from armulator.armv8.opcodes.opcode import Opcode


class AddSubWithCarry(Opcode):
    """
    ADC/ADCS/SBC/SBCS - the multi-precision arithmetic forms that thread the carry flag
    from one instruction to the next.
    """

    def __init__(self, instruction, d, n, m, setflags, sub_op, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.setflags = setflags
        self.sub_op = sub_op
        self.datasize = datasize

    def execute(self, processor):
        operand1 = processor.registers.get_x(self.n, self.datasize)
        operand2 = processor.registers.get_x(self.m, self.datasize)
        if self.sub_op:
            operand2 = bit_not(operand2, self.datasize)
        result, carry, overflow = add_with_carry(
            operand1, operand2, processor.registers.pstate.c, self.datasize
        )
        if self.setflags:
            processor.registers.set_flags(result, carry, overflow, self.datasize)
        processor.registers.set_x(self.d, result, self.datasize)
