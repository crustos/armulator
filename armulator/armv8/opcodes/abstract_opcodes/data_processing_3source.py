from armulator.armv8.bits_ops import lower_chunk, to_signed, to_unsigned
from armulator.armv8.opcodes.opcode import Opcode


class DataProcessing3Source(Opcode):
    """
    MADD, MSUB and the widening and high-half multiplies. MUL, MNEG, SMULL and UMULL are
    all aliases in which the accumulator is the zero register.
    """

    def __init__(self, instruction, d, n, m, a, operation, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.a = a
        self.operation = operation
        self.datasize = datasize

    def execute(self, processor):
        registers = processor.registers

        if self.operation in ('madd', 'msub'):
            operand1 = registers.get_x(self.n, self.datasize)
            operand2 = registers.get_x(self.m, self.datasize)
            addend = registers.get_x(self.a, self.datasize)
            product = operand1 * operand2
            result = addend + product if self.operation == 'madd' else addend - product
            registers.set_x(self.d, lower_chunk(result, self.datasize), self.datasize)
            return

        if self.operation in ('smaddl', 'smsubl', 'umaddl', 'umsubl'):
            # The sources are 32-bit; the accumulator and result are 64-bit.
            signed = self.operation.startswith('s')
            operand1 = registers.get_x(self.n, 32)
            operand2 = registers.get_x(self.m, 32)
            if signed:
                operand1 = to_signed(operand1, 32)
                operand2 = to_signed(operand2, 32)
            addend = to_signed(registers.get_x(self.a, 64), 64)
            product = operand1 * operand2
            result = addend + product if 'add' in self.operation else addend - product
            registers.set_x(self.d, to_unsigned(result, 64), 64)
            return

        # SMULH and UMULH return the upper 64 bits of a 128-bit product.
        operand1 = registers.get_x(self.n, 64)
        operand2 = registers.get_x(self.m, 64)
        if self.operation == 'smulh':
            product = to_signed(operand1, 64) * to_signed(operand2, 64)
        else:
            product = operand1 * operand2
        registers.set_x(self.d, to_unsigned(product >> 64, 64), 64)
