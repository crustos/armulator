from armulator.armv8.bits_ops import (
    big_endian_reverse,
    chain,
    count_leading_sign_bits,
    count_leading_zeros,
    reverse_bits,
    substring,
)
from armulator.armv8.opcodes.opcode import Opcode


class DataProcessing1Source(Opcode):
    """
    RBIT, REV16, REV32, REV, CLZ and CLS.

    The byte reversals operate within containers of a fixed width, so REV16 on a 64-bit
    register reverses four independent halfwords rather than the whole register.
    """

    def __init__(self, instruction, d, n, operation, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.operation = operation
        self.datasize = datasize

    def _reverse_containers(self, value, container_bits):
        """Reverse the bytes within each container of the given width."""
        container_bytes = container_bits // 8
        result = 0
        for index in range(self.datasize // container_bits):
            low = index * container_bits
            container = substring(value, low + container_bits - 1, low)
            result |= big_endian_reverse(container, container_bytes) << low
        return result

    def execute(self, processor):
        operand = processor.registers.get_x(self.n, self.datasize)

        if self.operation == 'rbit':
            result = reverse_bits(operand, self.datasize)
        elif self.operation == 'rev16':
            result = self._reverse_containers(operand, 16)
        elif self.operation == 'rev32':
            result = self._reverse_containers(operand, 32)
        elif self.operation == 'rev':
            result = big_endian_reverse(operand, self.datasize // 8)
        elif self.operation == 'clz':
            result = count_leading_zeros(operand, self.datasize)
        else:
            result = count_leading_sign_bits(operand, self.datasize)

        processor.registers.set_x(self.d, result, self.datasize)
