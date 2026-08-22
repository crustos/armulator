from armulator.armv8 import fp_ops
from armulator.armv8.bits_ops import (
    add,
    bit_not,
    lower_chunk,
    ones,
    sub,
    to_signed,
    to_unsigned,
)


def _signed_saturate(value, size):
    """Clamp to the signed range of ``size`` bits rather than wrapping."""
    high = (1 << (size - 1)) - 1
    low = -(1 << (size - 1))
    return to_unsigned(max(low, min(high, value)), size)


def _unsigned_saturate(value, size):
    return max(0, min((1 << size) - 1, value))
from armulator.armv8.opcodes.opcode import Opcode


class SimdThreeSame(Opcode):
    """
    Advanced SIMD operations taking two vectors of the same shape.

    Every lane is processed independently and the results reassembled, so the work here
    is a loop over element positions. A 64-bit (``8B``/``4H``/``2S``) operation writes
    only the low half of the register and zeroes the top, exactly as a D-register write
    does.
    """

    INTEGER_OPS = {
        'add': lambda a, b, size: add(a, b, size),
        'sub': lambda a, b, size: sub(a, b, size),
        'cmeq': lambda a, b, size: ones(size) if a == b else 0,
        'cmgt': lambda a, b, size: ones(size) if to_signed(a, size) > to_signed(b, size) else 0,
        'cmge': lambda a, b, size: ones(size) if to_signed(a, size) >= to_signed(b, size) else 0,
        'cmhi': lambda a, b, size: ones(size) if a > b else 0,
        'cmhs': lambda a, b, size: ones(size) if a >= b else 0,
        'umax': lambda a, b, size: max(a, b),
        'umin': lambda a, b, size: min(a, b),
        'smax': lambda a, b, size: to_unsigned(max(to_signed(a, size), to_signed(b, size)), size),
        'smin': lambda a, b, size: to_unsigned(min(to_signed(a, size), to_signed(b, size)), size),
        'mul': lambda a, b, size: lower_chunk(a * b, size),
        # Saturating forms clamp at the limit instead of wrapping round, which is what
        # makes them usable for signal data: an overflowing sample pins at full scale
        # rather than flipping sign.
        'sqadd': lambda a, b, size: _signed_saturate(
            to_signed(a, size) + to_signed(b, size), size),
        'sqsub': lambda a, b, size: _signed_saturate(
            to_signed(a, size) - to_signed(b, size), size),
        'uqadd': lambda a, b, size: _unsigned_saturate(a + b, size),
        'uqsub': lambda a, b, size: _unsigned_saturate(a - b, size),
    }

    FLOAT_OPS = {
        'fadd': fp_ops.fp_add,
        'fsub': fp_ops.fp_sub,
        'fmul': fp_ops.fp_mul,
        'fdiv': fp_ops.fp_div,
        'fmax': fp_ops.fp_max,
        'fmin': fp_ops.fp_min,
    }

    def __init__(self, instruction, d, n, m, operation, element_size, elements, datasize,
                 is_float=False):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.operation = operation
        self.element_size = element_size
        self.elements = elements
        self.datasize = datasize
        self.is_float = is_float

    def execute(self, processor):
        processor.check_fp_enabled()
        registers = processor.registers
        operation = (self.FLOAT_OPS if self.is_float else self.INTEGER_OPS)[self.operation]

        result = 0
        for index in range(self.elements):
            a = registers.get_v_element(self.n, index, self.element_size)
            b = registers.get_v_element(self.m, index, self.element_size)
            result |= lower_chunk(operation(a, b, self.element_size),
                                  self.element_size) << (index * self.element_size)

        registers.set_v(self.d, result, self.datasize)


class SimdBitwise(Opcode):
    """
    The bitwise vector operations AND, ORR, EOR, BIC and ORN, plus the MOV alias that
    ORR provides when both sources are the same register.

    These need no lane loop: the whole register is one operand.
    """

    def __init__(self, instruction, d, n, m, operation, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.operation = operation
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        operand1 = processor.registers.get_v(self.n, self.datasize)
        operand2 = processor.registers.get_v(self.m, self.datasize)

        if self.operation == 'and':
            result = operand1 & operand2
        elif self.operation == 'orr':
            result = operand1 | operand2
        elif self.operation == 'eor':
            result = operand1 ^ operand2
        elif self.operation == 'bic':
            result = operand1 & bit_not(operand2, self.datasize)
        else:
            result = operand1 | bit_not(operand2, self.datasize)

        processor.registers.set_v(self.d, result, self.datasize)
