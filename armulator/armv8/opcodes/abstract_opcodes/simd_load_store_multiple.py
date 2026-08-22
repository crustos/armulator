from armulator.armv8.bits_ops import lower_chunk
from armulator.armv8.enums import MemOp
from armulator.armv8.opcodes.opcode import Opcode


class SimdLoadStoreMultiple(Opcode):
    """
    LD1-LD4 and ST1-ST4 (multiple structures).

    These move an array of structures between memory and a set of registers, splitting
    the structure members apart on the way. ``LD2`` reading eight words puts the even
    ones in the first register and the odd ones in the second, which turns an
    interleaved array of pairs into two parallel vectors ready to be worked on.

    One loop covers all of them, parameterised two ways:

    ``selem``
        Members per structure - 1 for LD1, up to 4 for LD4. This is the de-interleaving
        stride: member *s* of every structure lands in register *t+s*.
    ``rpt``
        How many times the whole pattern repeats. Only LD1 uses this, to move two, three
        or four whole registers of contiguous data; for LD2-LD4 it is always 1.

    With ``selem`` of 1 the inner loop degenerates to a contiguous copy, which is exactly
    what LD1 should be - so the general form costs nothing in fidelity for the common
    case. Register numbers wrap at 32, so a transfer starting at V31 continues at V0.
    """

    def __init__(self, instruction, t, n, m, rpt, selem, element_size, elements,
                 memop, datasize, wback):
        super().__init__(instruction)
        self.t = t
        self.n = n
        self.m = m
        self.rpt = rpt
        self.selem = selem
        self.element_size = element_size
        self.elements = elements
        self.memop = memop
        self.datasize = datasize
        self.wback = wback

    @property
    def transfer_bytes(self):
        """Total bytes moved, which is also how far a post-index advances the base."""
        return self.rpt * self.selem * self.datasize // 8

    def execute(self, processor):
        processor.check_fp_enabled()

        if self.n == 31:
            processor.check_sp_alignment()
            address = processor.registers.get_sp()
        else:
            address = processor.registers.get_x(self.n)

        element_bytes = self.element_size // 8
        offset = 0
        for repeat in range(self.rpt):
            for element in range(self.elements):
                for member in range(self.selem):
                    register = (self.t + repeat * self.selem + member) % 32
                    location = lower_chunk(address + offset, 64)
                    if self.memop == MemOp.STORE:
                        processor.mem_set(
                            location, element_bytes,
                            processor.registers.get_v_element(
                                register, element, self.element_size),
                        )
                    else:
                        processor.registers.set_v_element(
                            register, element, self.element_size,
                            processor.mem_get(location, element_bytes),
                        )
                    offset += element_bytes

        if self.memop == MemOp.LOAD:
            # A 64-bit transfer writes the low half of each register and must clear the
            # top, the same rule as any other narrow write to a vector register. The
            # element writes above deliberately preserve it, so it is done here.
            if self.datasize == 64:
                for index in range(self.rpt * self.selem):
                    register = (self.t + index) % 32
                    processor.registers.set_v(
                        register, processor.registers.get_v(register, 64), 64)

        if self.wback:
            # Rm of 31 means the immediate form, which advances by the whole transfer.
            if self.m == 31:
                increment = self.transfer_bytes
            else:
                increment = processor.registers.get_x(self.m)
            address = lower_chunk(address + increment, 64)
            if self.n == 31:
                processor.registers.set_sp(address)
            else:
                processor.registers.set_x(self.n, address)
