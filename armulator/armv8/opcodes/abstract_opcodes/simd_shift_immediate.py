from armulator.armv8.bits_ops import asr, lower_chunk, lsl, lsr
from armulator.armv8.opcodes.opcode import Opcode


class SimdShiftImmediate(Opcode):
    """
    SHL, SSHR and USHR - shift every lane by a constant.

    The scalar forms are the same operation with a single lane, which is how they are
    modelled here. GCC emits scalar USHR to pull the upper half out of a packed pair of
    floats, so these appear in ordinary C that never mentions vectors.
    """

    def __init__(self, instruction, d, n, shift, operation, element_size, elements, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.shift = shift
        self.operation = operation
        self.element_size = element_size
        self.elements = elements
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        registers = processor.registers

        result = 0
        for index in range(self.elements):
            element = registers.get_v_element(self.n, index, self.element_size)
            if self.operation == 'shl':
                value = lsl(element, self.element_size, self.shift)
            elif self.operation == 'ushr':
                value = lsr(element, self.element_size, self.shift)
            else:
                value = asr(element, self.element_size, self.shift)
            result |= lower_chunk(value, self.element_size) << (index * self.element_size)

        registers.set_v(self.d, result, self.datasize)
