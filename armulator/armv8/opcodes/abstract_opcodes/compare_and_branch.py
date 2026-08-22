from armulator.armv8.bits_ops import lower_chunk
from armulator.armv8.opcodes.opcode import Opcode


class CompareAndBranch(Opcode):
    """
    CBZ and CBNZ - branch on a register being zero or non-zero, without touching the
    condition flags.
    """

    def __init__(self, instruction, t, offset, branch_if_nonzero, datasize):
        super().__init__(instruction)
        self.t = t
        self.offset = offset
        self.branch_if_nonzero = branch_if_nonzero
        self.datasize = datasize

    def execute(self, processor):
        operand = processor.registers.get_x(self.t, self.datasize)
        if bool(operand) == bool(self.branch_if_nonzero):
            pc = processor.registers.get_pc()
            processor.branch_to(lower_chunk(pc + self.offset, 64))
