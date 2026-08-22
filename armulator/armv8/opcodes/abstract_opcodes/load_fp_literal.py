from armulator.armv8.bits_ops import lower_chunk
from armulator.armv8.opcodes.abstract_opcodes.load_store_fp_immediate import _load
from armulator.armv8.opcodes.opcode import Opcode


class LoadFpLiteral(Opcode):
    """
    LDR (literal) into a SIMD or floating point register - how a compiler materialises a
    floating point constant that will not fit in an FMOV immediate.
    """

    def __init__(self, instruction, t, offset, datasize):
        super().__init__(instruction)
        self.t = t
        self.offset = offset
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        address = lower_chunk(processor.registers.get_pc() + self.offset, 64)
        processor.registers.set_v(
            self.t, _load(processor, address, self.datasize // 8), self.datasize
        )
