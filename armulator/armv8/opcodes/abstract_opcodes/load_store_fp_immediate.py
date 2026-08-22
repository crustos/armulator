from armulator.armv8.bits_ops import lower_chunk, substring
from armulator.armv8.enums import MemOp
from armulator.armv8.opcodes.opcode import Opcode


class LoadStoreFpImmediate(Opcode):
    """
    LDR/STR on a SIMD or floating point register with an immediate offset.

    The access width runs from a single byte (B registers) up to 128 bits (Q registers).
    A 128-bit access is wider than the memory hub's largest unit, so it is issued as two
    64-bit halves, little-endian order.
    """

    def __init__(self, instruction, t, n, offset, memop, datasize, wback, postindex):
        super().__init__(instruction)
        self.t = t
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

        if self.memop == MemOp.STORE:
            data = processor.registers.get_v(self.t, self.datasize)
            _store(processor, address, self.size, data)
        else:
            processor.registers.set_v(
                self.t, _load(processor, address, self.size), self.datasize
            )

        if self.wback:
            if self.postindex:
                address = lower_chunk(address + self.offset, 64)
            if self.n == 31:
                processor.registers.set_sp(address)
            else:
                processor.registers.set_x(self.n, address)


def _load(processor, address, size):
    """Read ``size`` bytes, splitting a 128-bit access into two 64-bit halves."""
    if size <= 8:
        return processor.mem_get(address, size)
    low = processor.mem_get(address, 8)
    high = processor.mem_get(lower_chunk(address + 8, 64), 8)
    return (high << 64) | low


def _store(processor, address, size, value):
    if size <= 8:
        processor.mem_set(address, size, value)
        return
    processor.mem_set(address, 8, lower_chunk(value, 64))
    processor.mem_set(lower_chunk(address + 8, 64), 8, substring(value, 127, 64))
