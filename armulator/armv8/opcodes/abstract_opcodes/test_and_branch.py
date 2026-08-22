from armulator.armv8.bits_ops import bit_at, lower_chunk
from armulator.armv8.opcodes.opcode import Opcode


class TestAndBranch(Opcode):
    """
    TBZ and TBNZ - branch on the state of a single bit. The bit number is split across
    the encoding, which is why a 32-bit form can still only name bits 0 to 31.
    """

    def __init__(self, instruction, t, bit_pos, offset, branch_if_set, datasize):
        super().__init__(instruction)
        self.t = t
        self.bit_pos = bit_pos
        self.offset = offset
        self.branch_if_set = branch_if_set
        self.datasize = datasize

    def execute(self, processor):
        operand = processor.registers.get_x(self.t, self.datasize)
        if bit_at(operand, self.bit_pos) == self.branch_if_set:
            pc = processor.registers.get_pc()
            processor.branch_to(lower_chunk(pc + self.offset, 64))
