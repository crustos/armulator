from armulator.armv8.bits_ops import align, lower_chunk
from armulator.armv8.opcodes.opcode import Opcode


class PcRelAddressing(Opcode):
    """
    ADR and ADRP. ADRP aligns the PC to a 4KB page before adding the (page-scaled) offset,
    which is how the toolchain builds a +/-4GB PC-relative address in two instructions.
    """

    def __init__(self, instruction, d, imm, page):
        super().__init__(instruction)
        self.d = d
        self.imm = imm
        self.page = page

    def execute(self, processor):
        base = processor.registers.get_pc()
        if self.page:
            base = align(base, 4096)
        processor.registers.set_x(self.d, lower_chunk(base + self.imm, 64))
