from armulator.armv8.bits_ops import lower_chunk
from armulator.armv8.opcodes.opcode import Opcode


class BranchConditional(Opcode):
    """
    B.cond - a PC-relative branch taken only when the condition holds. Unlike AArch32,
    this is the only conditional branch form; ordinary instructions carry no condition.
    """

    def __init__(self, instruction, offset, condition):
        super().__init__(instruction)
        self.offset = offset
        self.condition = condition

    def execute(self, processor):
        if processor.registers.condition_holds(self.condition):
            pc = processor.registers.get_pc()
            processor.branch_to(lower_chunk(pc + self.offset, 64))
