from armulator.armv8.bits_ops import lower_chunk
from armulator.armv8.opcodes.opcode import Opcode


class BranchImmediate(Opcode):
    """
    B and BL. The offset is relative to the address of the branch itself, and BL stores
    the address of the following instruction in X30.
    """

    def __init__(self, instruction, offset, with_link):
        super().__init__(instruction)
        self.offset = offset
        self.with_link = with_link

    def execute(self, processor):
        pc = processor.registers.get_pc()
        if self.with_link:
            processor.registers.set_lr(lower_chunk(pc + 4, 64))
        processor.branch_to(lower_chunk(pc + self.offset, 64))
