from armulator.armv8.bits_ops import lower_chunk, sign_extend
from armulator.armv8.enums import MemOp
from armulator.armv8.opcodes.opcode import Opcode


class LoadStorePair(Opcode):
    """
    LDP and STP - move two registers in one instruction. This is how AArch64 code saves
    and restores frames, so almost any real function prologue depends on it.
    """

    def __init__(self, instruction, t, t2, n, offset, memop, signed, datasize,
                 wback, postindex):
        super().__init__(instruction)
        self.t = t
        self.t2 = t2
        self.n = n
        self.offset = offset
        self.memop = memop
        self.signed = signed
        self.datasize = datasize
        self.wback = wback
        self.postindex = postindex

    @property
    def size(self):
        return self.datasize // 8

    def execute(self, processor):
        if self.n == 31:
            processor.check_sp_alignment()
            address = processor.registers.get_sp()
        else:
            address = processor.registers.get_x(self.n)

        if not self.postindex:
            address = lower_chunk(address + self.offset, 64)

        size = self.size
        if self.memop == MemOp.STORE:
            processor.mem_set(address, size,
                              processor.registers.get_x(self.t, self.datasize))
            processor.mem_set(lower_chunk(address + size, 64), size,
                              processor.registers.get_x(self.t2, self.datasize))
        else:
            data1 = processor.mem_get(address, size)
            data2 = processor.mem_get(lower_chunk(address + size, 64), size)
            if self.signed:
                # LDPSW sign extends each word into a 64-bit register.
                data1 = sign_extend(data1, 32, 64)
                data2 = sign_extend(data2, 32, 64)
                processor.registers.set_x(self.t, data1, 64)
                processor.registers.set_x(self.t2, data2, 64)
            else:
                processor.registers.set_x(self.t, data1, self.datasize)
                processor.registers.set_x(self.t2, data2, self.datasize)
                processor.note_deferred_load(self.t, address, size,
                                             datasize=self.datasize,
                                             regsize=self.datasize)
                processor.note_deferred_load(self.t2, lower_chunk(address + size, 64),
                                             size, datasize=self.datasize,
                                             regsize=self.datasize)

        if self.wback:
            if self.postindex:
                address = lower_chunk(address + self.offset, 64)
            if self.n == 31:
                processor.registers.set_sp(address)
            else:
                processor.registers.set_x(self.n, address)
