from armulator.armv8 import fp_ops
from armulator.armv8.bits_ops import lower_chunk
from armulator.armv8.opcodes.opcode import Opcode


class SimdByElement(Opcode):
    """
    FMUL, FMLA and FMLS by element - multiply every lane of one vector by a single
    chosen lane of another.

    This is what a compiler emits when it vectorises ``array[i] * scalar``: the scalar is
    broadcast implicitly by the instruction rather than by a separate DUP.
    """

    def __init__(self, instruction, d, n, m, index, operation, element_size, elements,
                 datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.index = index
        self.operation = operation
        self.element_size = element_size
        self.elements = elements
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        registers = processor.registers
        size = self.element_size
        scalar = registers.get_v_element(self.m, self.index, size)

        result = 0
        for index in range(self.elements):
            element = registers.get_v_element(self.n, index, size)
            product = fp_ops.fp_mul(element, scalar, size)
            if self.operation == 'fmul':
                value = product
            else:
                accumulator = registers.get_v_element(self.d, index, size)
                if self.operation == 'fmls':
                    product = fp_ops.fp_neg(product, size)
                value = fp_ops.fp_add(accumulator, product, size)
            result |= lower_chunk(value, size) << (index * size)

        registers.set_v(self.d, result, self.datasize)
