from armulator.armv8.bits_ops import decode_reg_extend, lower_chunk
from armulator.armv8.enums import MemOp
from armulator.armv8.opcodes.abstract_opcodes.load_store_fp_immediate import _load, _store
from armulator.armv8.opcodes.opcode import Opcode


class LoadStoreFpRegisterOffset(Opcode):
    """
    LDR/STR on a SIMD or floating point register with a register offset, the form used
    when walking an array of vectors.
    """

    def __init__(self, instruction, t, n, m, extend_type, shift, memop, datasize):
        super().__init__(instruction)
        self.t = t
        self.n = n
        self.m = m
        self.extend_type = extend_type
        self.shift = shift
        self.memop = memop
        self.datasize = datasize

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

        offset = decode_reg_extend(
            processor.registers.get_x(self.m), self.extend_type, self.shift, 64
        )
        address = lower_chunk(address + offset, 64)

        if self.memop == MemOp.STORE:
            _store(processor, address, self.size,
                   processor.registers.get_v(self.t, self.datasize))
        else:
            processor.registers.set_v(
                self.t, _load(processor, address, self.size), self.datasize
            )
