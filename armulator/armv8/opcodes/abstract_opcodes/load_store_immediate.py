from armulator.armv8.bits_ops import lower_chunk, sign_extend
from armulator.armv8.enums import MemOp
from armulator.armv8.opcodes.opcode import Opcode


class LoadStoreImmediate(Opcode):
    """
    LDR/STR and their sized and signed variants with an immediate offset.

    One implementation covers four encodings, which differ only in how the address is
    formed and whether the base is written back:

      * unsigned offset  - scaled immediate, no writeback
      * unscaled (LDUR)  - signed 9-bit immediate, no writeback
      * post-index       - access at the base, then base += offset
      * pre-index        - base += offset, then access there

    Register 31 means SP for the base, so the stack alignment check applies to it.
    """

    def __init__(self, instruction, t, n, offset, memop, signed, regsize, datasize,
                 wback, postindex):
        super().__init__(instruction)
        self.t = t
        self.n = n
        self.offset = offset
        self.memop = memop
        self.signed = signed
        self.regsize = regsize
        self.datasize = datasize
        self.wback = wback
        self.postindex = postindex

    @property
    def size(self):
        """Access width in bytes."""
        return self.datasize // 8

    def execute(self, processor):
        if self.n == 31:
            processor.check_sp_alignment()
            address = processor.registers.get_sp()
        else:
            address = processor.registers.get_x(self.n)

        if not self.postindex:
            address = lower_chunk(address + self.offset, 64)

        if self.memop == MemOp.STORE:
            data = processor.registers.get_x(self.t, self.regsize)
            processor.mem_set(address, self.size, data)
        elif self.memop == MemOp.LOAD:
            data = processor.mem_get(address, self.size)
            if self.signed:
                data = sign_extend(data, self.datasize, self.regsize)
            processor.registers.set_x(self.t, data, self.regsize)
            # The value may be settled again later; see ArmV8.note_deferred_load.
            processor.note_deferred_load(self.t, address, self.size, self.signed,
                                         self.datasize, self.regsize)
        # A prefetch has no architectural effect in this model.

        if self.wback:
            if self.postindex:
                address = lower_chunk(address + self.offset, 64)
            if self.n == 31:
                processor.registers.set_sp(address)
            else:
                processor.registers.set_x(self.n, address)
