from armulator.armv8.bits_ops import lower_chunk, replicate, sign_extend
from armulator.armv8.opcodes.opcode import Opcode


class SimdImmediate(Opcode):
    """
    MOVI, MVNI and the vector FMOV immediate - how a vector constant is materialised
    without a memory load. The decoder has already expanded the encoding into the full
    64-bit pattern that gets replicated across the register.
    """

    def __init__(self, instruction, d, imm64, datasize, invert=False):
        super().__init__(instruction)
        self.d = d
        self.imm64 = imm64
        self.datasize = datasize
        self.invert = invert

    def execute(self, processor):
        processor.check_fp_enabled()
        value = self.imm64
        if self.invert:
            value = ~value & 0xFFFFFFFFFFFFFFFF
        if self.datasize == 128:
            value = (value << 64) | value
        processor.registers.set_v(self.d, value, self.datasize)


class SimdDuplicate(Opcode):
    """
    DUP - broadcast one value into every lane, either from a general purpose register
    or from a lane of another vector.
    """

    def __init__(self, instruction, d, n, index, element_size, elements, datasize,
                 from_general):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.index = index
        self.element_size = element_size
        self.elements = elements
        self.datasize = datasize
        self.from_general = from_general

    def execute(self, processor):
        processor.check_fp_enabled()
        if self.from_general:
            element = processor.registers.get_x(self.n, self.element_size)
        else:
            element = processor.registers.get_v_element(self.n, self.index, self.element_size)
        processor.registers.set_v(
            self.d, replicate(element, self.element_size, self.datasize), self.datasize
        )


class SimdInsert(Opcode):
    """
    INS - write a single lane, from a general purpose register or from a lane of another
    vector. This is the one vector write that must preserve the rest of the register.
    """

    def __init__(self, instruction, d, n, dst_index, src_index, element_size, from_general):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.dst_index = dst_index
        self.src_index = src_index
        self.element_size = element_size
        self.from_general = from_general

    def execute(self, processor):
        processor.check_fp_enabled()
        if self.from_general:
            value = processor.registers.get_x(self.n, self.element_size)
        else:
            value = processor.registers.get_v_element(
                self.n, self.src_index, self.element_size
            )
        processor.registers.set_v_element(self.d, self.dst_index, self.element_size, value)


class SimdExtract(Opcode):
    """
    UMOV and SMOV - move one lane into a general purpose register, zero or sign extended.
    """

    def __init__(self, instruction, d, n, index, element_size, regsize, signed):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.index = index
        self.element_size = element_size
        self.regsize = regsize
        self.signed = signed

    def execute(self, processor):
        processor.check_fp_enabled()
        value = processor.registers.get_v_element(self.n, self.index, self.element_size)
        if self.signed:
            value = sign_extend(value, self.element_size, self.regsize)
        processor.registers.set_x(self.d, lower_chunk(value, self.regsize), self.regsize)
