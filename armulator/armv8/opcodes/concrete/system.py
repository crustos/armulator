"""
Concrete encodings for exception generation and the system instruction space.
"""

from armulator.armv8.bits_ops import bit_at, chain, substring
from armulator.armv8.enums import PSTATEField
from armulator.armv8.opcodes.abstract_opcodes.barriers import Barrier
from armulator.armv8.opcodes.abstract_opcodes.exception_generation import ExceptionGeneration
from armulator.armv8.opcodes.abstract_opcodes.hints import Hint
from armulator.armv8.opcodes.abstract_opcodes.msr_immediate import MsrImmediate
from armulator.armv8.opcodes.abstract_opcodes.system_instruction import SystemInstruction
from armulator.armv8.opcodes.abstract_opcodes.system_register_move import SystemRegisterMove

#: (opc, LL) -> which exception the instruction raises.
_EXCEPTIONS = {
    (0b000, 0b01): 'svc',
    (0b000, 0b10): 'hvc',
    (0b000, 0b11): 'smc',
    (0b001, 0b00): 'brk',
    (0b010, 0b00): 'hlt',
}

#: (CRm, op2) -> hint. Everything else in the hint space executes as NOP.
_HINTS = {
    (0b0000, 0b000): 'nop',
    (0b0000, 0b001): 'yield',
    (0b0000, 0b010): 'wfe',
    (0b0000, 0b011): 'wfi',
    (0b0000, 0b100): 'sev',
    (0b0000, 0b101): 'sevl',
}

_BARRIERS = {0b100: 'dsb', 0b101: 'dmb', 0b110: 'isb'}

#: (op1, op2) -> which PSTATE field MSR (immediate) writes.
_PSTATE_FIELDS = {
    (0b000, 0b101): PSTATEField.SP,
    (0b011, 0b110): PSTATEField.DAIFSET,
    (0b011, 0b111): PSTATEField.DAIFCLR,
}


class ExceptionGenerationA64(ExceptionGeneration):
    """
    1 1 0 1 0 1 0 0 opc imm16 op2 LL
    """

    @staticmethod
    def from_bitarray(instr, processor):
        opc = substring(instr, 23, 21)
        ll = substring(instr, 1, 0)
        if substring(instr, 4, 2) != 0b000:
            return None
        kind = _EXCEPTIONS.get((opc, ll))
        if kind is None:
            return None
        return ExceptionGenerationA64(instr, imm16=substring(instr, 20, 5), kind=kind)


class HintA64(Hint):
    """
    1 1 0 1 0 1 0 1 0 0 0 0 0 0 1 1 0 0 1 0 CRm op2 1 1 1 1 1
    """

    @staticmethod
    def from_bitarray(instr, processor):
        crm = substring(instr, 11, 8)
        op2 = substring(instr, 7, 5)
        # Unallocated hints are required to behave as NOP rather than fault.
        return HintA64(instr, kind=_HINTS.get((crm, op2), 'nop'))


class ClrexA64(Hint):
    """
    CLREX - drop this core's exclusive reservation.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        return ClrexA64(instr, kind='clrex')


class BarrierA64(Barrier):
    """
    1 1 0 1 0 1 0 1 0 0 0 0 0 0 1 1 0 0 1 1 CRm op2 Rt
    """

    @staticmethod
    def from_bitarray(instr, processor):
        op2 = substring(instr, 7, 5)
        kind = _BARRIERS.get(op2)
        if kind is None:
            return None
        if kind == 'isb' and substring(instr, 4, 0) != 0b11111:
            return None
        return BarrierA64(instr, kind=kind, crm=substring(instr, 11, 8))


class MsrImmediateA64(MsrImmediate):
    """
    1 1 0 1 0 1 0 1 0 0 0 0 0 op1 0 1 0 0 CRm op2 1 1 1 1 1
    """

    @staticmethod
    def from_bitarray(instr, processor):
        op1 = substring(instr, 18, 16)
        op2 = substring(instr, 7, 5)
        field = _PSTATE_FIELDS.get((op1, op2))
        if field is None:
            return None
        if substring(instr, 4, 0) != 0b11111:
            return None
        return MsrImmediateA64(instr, field=field, operand=substring(instr, 11, 8))


class SystemRegisterMoveA64(SystemRegisterMove):
    """
    1 1 0 1 0 1 0 1 0 0 L 1 op0 op1 CRn CRm op2 Rt - MRS when L is 1, MSR when 0.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        return SystemRegisterMoveA64(
            instr,
            t=substring(instr, 4, 0),
            # op0 is encoded in two bits with the top bit implied set.
            op0=chain(1, bit_at(instr, 19), 1),
            op1=substring(instr, 18, 16),
            crn=substring(instr, 15, 12),
            crm=substring(instr, 11, 8),
            op2=substring(instr, 7, 5),
            read=bit_at(instr, 21),
        )


class SystemInstructionA64(SystemInstruction):
    """
    1 1 0 1 0 1 0 1 0 0 L 0 1 op1 CRn CRm op2 Rt - SYS and SYSL.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        return SystemInstructionA64(
            instr,
            t=substring(instr, 4, 0),
            op1=substring(instr, 18, 16),
            crn=substring(instr, 15, 12),
            crm=substring(instr, 11, 8),
            op2=substring(instr, 7, 5),
            read=bit_at(instr, 21),
        )
