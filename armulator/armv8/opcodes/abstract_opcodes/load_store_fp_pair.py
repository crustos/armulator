from armulator.armv8.bits_ops import lower_chunk
from armulator.armv8.enums import MemOp
from armulator.armv8.opcodes.abstract_opcodes.load_store_fp_immediate import _load, _store
from armulator.armv8.opcodes.opcode import Opcode


class LoadStoreFpPair(Opcode):
    """
    LDP/STP on SIMD and floating point registers.

    The Q form moves 32 bytes in one instruction, which is why an optimised memcpy
    reaches for it, and why it is usually the first thing compiler output needs from a
    SIMD implementation.
    """

    def __init__(self, instruction, t, t2, n, offset, memop, datasize, wback, postindex):
        super().__init__(instruction)
        self.t = t
        self.t2 = t2
        self.n = n
        self.offset = offset
        self.memop = memop
        self.datasize = datasize
        self.wback = wback
        self.postindex = postindex

    @property
    def size(self):
        return self.datasize // 8

    def execute(self, processor):
        processor.check_fp_enabled()

        if self.n == 31:
            processor.check_sp_alignment()
            address = processor.registers.get_sp()
        else:
            address = processor.registers.get_x(self.n)

        if not self.postindex:
            address = lower_chunk(address + self.offset, 64)

        size = self.size
        second = lower_chunk(address + size, 64)
        if self.memop == MemOp.STORE:
            _store(processor, address, size, processor.registers.get_v(self.t, self.datasize))
            _store(processor, second, size, processor.registers.get_v(self.t2, self.datasize))
        else:
            processor.registers.set_v(self.t, _load(processor, address, size), self.datasize)
            processor.registers.set_v(self.t2, _load(processor, second, size), self.datasize)

        if self.wback:
            if self.postindex:
                address = lower_chunk(address + self.offset, 64)
            if self.n == 31:
                processor.registers.set_sp(address)
            else:
                processor.registers.set_x(self.n, address)
