from armulator.armv8.bits_ops import lower_chunk, to_signed, to_unsigned
from armulator.armv8.opcodes.opcode import Opcode


class SimdAcrossLanes(Opcode):
    """
    ADDV, SMAXV/UMAXV and SMINV/UMINV - reduce every lane of a vector to one value.

    The result is a scalar in the destination's lowest lane with everything above it
    cleared, so the answer can be moved straight out with UMOV or used as an address.
    """

    def __init__(self, instruction, d, n, operation, element_size, elements):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.operation = operation
        self.element_size = element_size
        self.elements = elements

    def execute(self, processor):
        processor.check_fp_enabled()
        registers = processor.registers
        size = self.element_size
        values = [registers.get_v_element(self.n, index, size)
                  for index in range(self.elements)]

        if self.operation == 'addv':
            # The sum is taken modulo the element width; the widening form is SADDLV.
            result = lower_chunk(sum(values), size)
        elif self.operation == 'umaxv':
            result = max(values)
        elif self.operation == 'uminv':
            result = min(values)
        elif self.operation == 'smaxv':
            result = to_unsigned(max(to_signed(v, size) for v in values), size)
        else:
            result = to_unsigned(min(to_signed(v, size) for v in values), size)

        registers.set_v(self.d, result, size)


class SimdPairwise(Opcode):
    """
    ADDP - add adjacent pairs of lanes.

    The two source vectors are treated as one long sequence, so the low half of the
    result comes from Vn and the high half from Vm. Repeatedly applying it is the usual
    way to fold a vector down to a single value.
    """

    def __init__(self, instruction, d, n, m, element_size, elements, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.element_size = element_size
        self.elements = elements
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        registers = processor.registers
        size = self.element_size

        concatenated = [registers.get_v_element(self.n, i, size)
                        for i in range(self.elements)]
        concatenated += [registers.get_v_element(self.m, i, size)
                         for i in range(self.elements)]

        result = 0
        for index in range(self.elements):
            total = concatenated[2 * index] + concatenated[2 * index + 1]
            result |= lower_chunk(total, size) << (index * size)

        registers.set_v(self.d, result, self.datasize)
