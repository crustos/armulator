from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.opcodes.abstract_opcodes.add_sub_immediate import AddSubImmediate


class AddSubImmediateA64(AddSubImmediate):
    """
    sf op S 1 0 0 0 1 0 sh imm12 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        rd = substring(instr, 4, 0)
        rn = substring(instr, 9, 5)
        imm12 = substring(instr, 21, 10)
        shift = substring(instr, 23, 22)
        setflags = bit_at(instr, 29)
        sub_op = bit_at(instr, 30)
        datasize = 64 if bit_at(instr, 31) else 32

        # Only a left shift of 0 or 12 is defined; the other two encodings are reserved.
        if shift == 0b00:
            imm = imm12
        elif shift == 0b01:
            imm = imm12 << 12
        else:
            return None

        return AddSubImmediateA64(instr, d=rd, n=rn, imm=imm, setflags=setflags,
                                  sub_op=sub_op, datasize=datasize)
