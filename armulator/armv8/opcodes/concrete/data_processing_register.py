"""
Concrete encodings for the data processing (register) group.
"""

from armulator.armv8.bits_ops import bit_at, substring
from armulator.armv8.enums import ExtendType, LogicalOp, ShiftType
from armulator.armv8.opcodes.abstract_opcodes.add_sub_extended_register import (
    AddSubExtendedRegister,
)
from armulator.armv8.opcodes.abstract_opcodes.add_sub_shifted_register import (
    AddSubShiftedRegister,
)
from armulator.armv8.opcodes.abstract_opcodes.add_sub_with_carry import AddSubWithCarry
from armulator.armv8.opcodes.abstract_opcodes.conditional_compare import ConditionalCompare
from armulator.armv8.opcodes.abstract_opcodes.conditional_select import ConditionalSelect
from armulator.armv8.opcodes.abstract_opcodes.data_processing_1source import (
    DataProcessing1Source,
)
from armulator.armv8.opcodes.abstract_opcodes.data_processing_2source import (
    DataProcessing2Source,
)
from armulator.armv8.opcodes.abstract_opcodes.data_processing_3source import (
    DataProcessing3Source,
)
from armulator.armv8.opcodes.abstract_opcodes.logical_shifted_register import (
    LogicalShiftedRegister,
)

_LOGICAL_OPS = {0b00: LogicalOp.AND, 0b01: LogicalOp.ORR,
                0b10: LogicalOp.EOR, 0b11: LogicalOp.AND}

_ONE_SOURCE = {0b000000: 'rbit', 0b000001: 'rev16', 0b000010: 'rev32',
               0b000011: 'rev', 0b000100: 'clz', 0b000101: 'cls'}

_TWO_SOURCE = {0b000010: 'udiv', 0b000011: 'sdiv', 0b001000: 'lslv',
               0b001001: 'lsrv', 0b001010: 'asrv', 0b001011: 'rorv'}

#: (op31, o0) -> which three-source multiply this is.
_THREE_SOURCE = {
    (0b000, 0): 'madd', (0b000, 1): 'msub',
    (0b001, 0): 'smaddl', (0b001, 1): 'smsubl',
    (0b010, 0): 'smulh',
    (0b101, 0): 'umaddl', (0b101, 1): 'umsubl',
    (0b110, 0): 'umulh',
}


class LogicalShiftedRegisterA64(LogicalShiftedRegister):
    """
    sf opc 0 1 0 1 0 shift N Rm imm6 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        sf = bit_at(instr, 31)
        datasize = 64 if sf else 32
        imm6 = substring(instr, 15, 10)
        # A 32-bit operation cannot shift by 32 or more.
        if not sf and bit_at(imm6, 5):
            return None
        opc = substring(instr, 30, 29)
        return LogicalShiftedRegisterA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), op=_LOGICAL_OPS[opc],
            shift_type=ShiftType(substring(instr, 23, 22)), shift_amount=imm6,
            invert=bit_at(instr, 21), setflags=(opc == 0b11), datasize=datasize,
        )


class AddSubShiftedRegisterA64(AddSubShiftedRegister):
    """
    sf op S 0 1 0 1 1 shift 0 Rm imm6 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        sf = bit_at(instr, 31)
        datasize = 64 if sf else 32
        shift = substring(instr, 23, 22)
        imm6 = substring(instr, 15, 10)
        # ROR is not a valid shift for add and subtract.
        if shift == 0b11:
            return None
        if not sf and bit_at(imm6, 5):
            return None
        return AddSubShiftedRegisterA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), shift_type=ShiftType(shift), shift_amount=imm6,
            setflags=bit_at(instr, 29), sub_op=bit_at(instr, 30), datasize=datasize,
        )


class AddSubExtendedRegisterA64(AddSubExtendedRegister):
    """
    sf op S 0 1 0 1 1 0 0 1 Rm option imm3 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        imm3 = substring(instr, 12, 10)
        # The extend shift amount is limited to 0..4.
        if imm3 > 4:
            return None
        if substring(instr, 23, 22) != 0b00:
            return None
        return AddSubExtendedRegisterA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16),
            extend_type=ExtendType(substring(instr, 15, 13)), shift=imm3,
            setflags=bit_at(instr, 29), sub_op=bit_at(instr, 30),
            datasize=64 if bit_at(instr, 31) else 32,
        )


class AddSubWithCarryA64(AddSubWithCarry):
    """
    sf op S 1 1 0 1 0 0 0 0 Rm 0 0 0 0 0 0 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        if substring(instr, 15, 10) != 0:
            return None
        return AddSubWithCarryA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), setflags=bit_at(instr, 29),
            sub_op=bit_at(instr, 30), datasize=64 if bit_at(instr, 31) else 32,
        )


class ConditionalSelectA64(ConditionalSelect):
    """
    sf op S 1 1 0 1 0 1 0 0 Rm cond op2 Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        op = bit_at(instr, 30)
        op2 = substring(instr, 11, 10)
        if bit_at(instr, 29) or bit_at(op2, 1):
            return None
        return ConditionalSelectA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), condition=substring(instr, 15, 12),
            else_invert=op, else_increment=bit_at(op2, 0),
            datasize=64 if bit_at(instr, 31) else 32,
        )


class ConditionalCompareA64(ConditionalCompare):
    """
    sf op 1 1 1 0 1 0 0 1 0 Rm/imm5 cond mode 0 Rn 0 nzcv

    ``mode`` at instr[11] chooses between comparing against a register and comparing
    against a five-bit immediate.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        if not bit_at(instr, 29) or bit_at(instr, 10) or bit_at(instr, 4):
            return None
        datasize = 64 if bit_at(instr, 31) else 32
        immediate_form = bit_at(instr, 11)
        if immediate_form:
            operand2 = substring(instr, 20, 16)
        else:
            m = substring(instr, 20, 16)
            operand2 = (lambda processor, m=m, size=datasize:
                        processor.registers.get_x(m, size))
        return ConditionalCompareA64(
            instr, n=substring(instr, 9, 5), operand2=operand2,
            condition=substring(instr, 15, 12), flags=substring(instr, 3, 0),
            sub_op=bit_at(instr, 30), datasize=datasize,
        )


class DataProcessing1SourceA64(DataProcessing1Source):
    """
    sf 1 S 1 1 0 1 0 1 1 0 opcode2 opcode Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        if bit_at(instr, 29) or substring(instr, 20, 16) != 0:
            return None
        opcode = substring(instr, 15, 10)
        operation = _ONE_SOURCE.get(opcode)
        if operation is None:
            return None
        sf = bit_at(instr, 31)
        # REV on a 32-bit register is encoded as REV32 would be on a 64-bit one.
        if not sf:
            if opcode == 0b000011:
                return None
            if opcode == 0b000010:
                operation = 'rev'
        return DataProcessing1SourceA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            operation=operation, datasize=64 if sf else 32,
        )


class DataProcessing2SourceA64(DataProcessing2Source):
    """
    sf 0 S 1 1 0 1 0 1 1 0 Rm opcode Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        if bit_at(instr, 29):
            return None
        operation = _TWO_SOURCE.get(substring(instr, 15, 10))
        if operation is None:
            return None
        return DataProcessing2SourceA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), operation=operation,
            datasize=64 if bit_at(instr, 31) else 32,
        )


class DataProcessing3SourceA64(DataProcessing3Source):
    """
    sf op54 1 1 0 1 1 op31 Rm o0 Ra Rn Rd
    """

    @staticmethod
    def from_bitarray(instr, processor):
        if substring(instr, 30, 29) != 0b00:
            return None
        sf = bit_at(instr, 31)
        operation = _THREE_SOURCE.get((substring(instr, 23, 21), bit_at(instr, 15)))
        if operation is None:
            return None
        # The widening and high-half multiplies only exist in the 64-bit encoding.
        if not sf and operation != 'madd' and operation != 'msub':
            return None
        return DataProcessing3SourceA64(
            instr, d=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), a=substring(instr, 14, 10),
            operation=operation, datasize=64 if sf else 32,
        )
