from armulator.armv8.bits_ops import (
    asr,
    lower_chunk,
    lsl,
    lsr,
    ror,
    to_signed,
    to_unsigned,
)
from armulator.armv8.opcodes.opcode import Opcode


class DataProcessing2Source(Opcode):
    """
    UDIV, SDIV and the variable shifts LSLV/LSRV/ASRV/RORV.

    Division by zero produces zero rather than trapping - AArch64 removed the divide
    trap that AArch32 had - and the shift amount is taken modulo the register width.
    """

    def __init__(self, instruction, d, n, m, operation, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.operation = operation
        self.datasize = datasize

    def execute(self, processor):
        operand1 = processor.registers.get_x(self.n, self.datasize)
        operand2 = processor.registers.get_x(self.m, self.datasize)

        if self.operation == 'udiv':
            result = 0 if operand2 == 0 else operand1 // operand2
        elif self.operation == 'sdiv':
            dividend = to_signed(operand1, self.datasize)
            divisor = to_signed(operand2, self.datasize)
            if divisor == 0:
                result = 0
            else:
                # Truncate toward zero, unlike python's floor division.
                quotient = abs(dividend) // abs(divisor)
                if (dividend < 0) != (divisor < 0):
                    quotient = -quotient
                result = to_unsigned(quotient, self.datasize)
        else:
            # The shift amount is the low bits of the second operand.
            amount = operand2 % self.datasize
            if self.operation == 'lslv':
                result = lsl(operand1, self.datasize, amount)
            elif self.operation == 'lsrv':
                result = lsr(operand1, self.datasize, amount)
            elif self.operation == 'asrv':
                result = asr(operand1, self.datasize, amount)
            else:
                result = ror(operand1, self.datasize, amount)

        processor.registers.set_x(self.d, lower_chunk(result, self.datasize), self.datasize)
