from armulator.armv8.bits_ops import bit_not, lower_chunk
from armulator.armv8.opcodes.opcode import Opcode


class ConditionalSelect(Opcode):
    """
    CSEL/CSINC/CSINV/CSNEG, and the CSET/CSETM/CINC aliases.

    These let a compiler turn a short branch into straight-line code, so they turn up
    constantly even in simple firmware.
    """

    def __init__(self, instruction, d, n, m, condition, else_invert, else_increment, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.condition = condition
        self.else_invert = else_invert
        self.else_increment = else_increment
        self.datasize = datasize

    def execute(self, processor):
        if processor.registers.condition_holds(self.condition):
            result = processor.registers.get_x(self.n, self.datasize)
        else:
            result = processor.registers.get_x(self.m, self.datasize)
            if self.else_invert:
                result = bit_not(result, self.datasize)
            if self.else_increment:
                result = lower_chunk(result + 1, self.datasize)
        processor.registers.set_x(self.d, result, self.datasize)
