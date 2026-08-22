from armulator.armv8.bits_ops import lower_chunk
from armulator.armv8.opcodes.opcode import Opcode


class BranchRegister(Opcode):
    """
    BR, BLR and RET. RET is just BR with X30 as the default source, but it is encoded
    separately so that a return is visible to branch prediction and to a disassembler.
    """

    def __init__(self, instruction, n, with_link):
        super().__init__(instruction)
        self.n = n
        self.with_link = with_link

    def execute(self, processor):
        target = processor.registers.get_x(self.n)
        if self.with_link:
            processor.registers.set_lr(lower_chunk(processor.registers.get_pc() + 4, 64))
        processor.branch_to(target)
