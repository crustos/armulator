from armulator.armv8.bits_ops import decode_reg_extend, lower_chunk, sign_extend
from armulator.armv8.enums import MemOp
from armulator.armv8.opcodes.opcode import Opcode


class LoadStoreRegisterOffset(Opcode):
    """
    LDR/STR with a register offset, optionally extended and scaled - the form a compiler
    emits for indexing an array. The offset register can be extended from 32 bits, which
    is why the extend type is part of the encoding.
    """

    def __init__(self, instruction, t, n, m, extend_type, shift, memop, signed,
                 regsize, datasize):
        super().__init__(instruction)
        self.t = t
        self.n = n
        self.m = m
        self.extend_type = extend_type
        self.shift = shift
        self.memop = memop
        self.signed = signed
        self.regsize = regsize
        self.datasize = datasize

    @property
    def size(self):
        return self.datasize // 8

    def execute(self, processor):
        if self.n == 31:
            processor.check_sp_alignment()
            address = processor.registers.get_sp()
        else:
            address = processor.registers.get_x(self.n)

        offset = decode_reg_extend(
            processor.registers.get_x(self.m), self.extend_type, self.shift, 64
        )
        address = lower_chunk(address + offset, 64)

        if self.memop == MemOp.STORE:
            processor.mem_set(address, self.size,
                              processor.registers.get_x(self.t, self.regsize))
        elif self.memop == MemOp.LOAD:
            data = processor.mem_get(address, self.size)
            if self.signed:
                data = sign_extend(data, self.datasize, self.regsize)
            processor.registers.set_x(self.t, data, self.regsize)
            processor.note_deferred_load(self.t, address, self.size, self.signed,
                                         self.datasize, self.regsize)
