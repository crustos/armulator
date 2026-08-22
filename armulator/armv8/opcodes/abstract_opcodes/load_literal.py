from armulator.armv8.bits_ops import lower_chunk, sign_extend
from armulator.armv8.opcodes.opcode import Opcode


class LoadLiteral(Opcode):
    """
    LDR (literal) - load from an address relative to the instruction itself, used for
    constant pools that are too wide to build with MOVZ/MOVK.
    """

    def __init__(self, instruction, t, offset, signed, datasize, is_prefetch=False):
        super().__init__(instruction)
        self.t = t
        self.offset = offset
        self.signed = signed
        self.datasize = datasize
        self.is_prefetch = is_prefetch

    def execute(self, processor):
        if self.is_prefetch:
            return
        address = lower_chunk(processor.registers.get_pc() + self.offset, 64)
        data = processor.mem_get(address, self.datasize // 8)
        if self.signed:
            processor.registers.set_x(self.t, sign_extend(data, 32, 64), 64)
        else:
            processor.registers.set_x(self.t, data, self.datasize)
