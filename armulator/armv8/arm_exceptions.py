"""
Exceptions used to unwind out of the middle of instruction execution.

These are python exceptions standing in for the architectural act of taking an exception:
raising one abandons the rest of the instruction, and the CPU model catches it in
emulate_cycle and vectors to the appropriate handler.
"""

from armulator.armv8.enums import ExceptionType


class ArmV8Exception(Exception):
    """
    Base for everything that diverts AArch64 instruction execution.
    """
    exception_type = ExceptionType.UNCATEGORIZED

    def __init__(self, message='', preferred_return=None):
        super().__init__(message)
        self.preferred_return = preferred_return


class EndOfInstruction(ArmV8Exception):
    """
    Not an architectural exception - abandons the instruction without vectoring,
    used where the pseudocode calls EndOfInstruction().
    """


class UndefinedInstructionException(ArmV8Exception):
    exception_type = ExceptionType.UNCATEGORIZED


class SVCException(ArmV8Exception):
    exception_type = ExceptionType.SUPERVISOR_CALL

    def __init__(self, immediate=0, message=''):
        super().__init__(message)
        self.immediate = immediate


class HVCException(ArmV8Exception):
    exception_type = ExceptionType.HYPERVISOR_CALL

    def __init__(self, immediate=0, message=''):
        super().__init__(message)
        self.immediate = immediate


class SMCException(ArmV8Exception):
    exception_type = ExceptionType.MONITOR_CALL

    def __init__(self, immediate=0, message=''):
        super().__init__(message)
        self.immediate = immediate


class SoftwareBreakpointException(ArmV8Exception):
    exception_type = ExceptionType.SOFTWARE_BREAKPOINT

    def __init__(self, immediate=0, message=''):
        super().__init__(message)
        self.immediate = immediate


class DataAbortException(ArmV8Exception):
    exception_type = ExceptionType.DATA_ABORT

    def __init__(self, address=0, is_write=False, message='', status=0b100001):
        super().__init__(message)
        self.address = address
        self.is_write = is_write
        #: DFSC for ESR_ELx.ISS. Defaults to an alignment fault, which is the only data
        #: abort the model raised before translation existed.
        self.status = status


class InstructionAbortException(ArmV8Exception):
    exception_type = ExceptionType.INSTRUCTION_ABORT

    def __init__(self, address=0, message='', status=0b000100):
        super().__init__(message)
        self.address = address
        #: IFSC for ESR_ELx.ISS.
        self.status = status


class PcAlignmentException(ArmV8Exception):
    exception_type = ExceptionType.PC_ALIGNMENT

    def __init__(self, address=0, message=''):
        super().__init__(message)
        self.address = address


class SpAlignmentException(ArmV8Exception):
    exception_type = ExceptionType.SP_ALIGNMENT


class IllegalStateException(ArmV8Exception):
    exception_type = ExceptionType.ILLEGAL_STATE


class SystemRegisterTrapException(ArmV8Exception):
    exception_type = ExceptionType.SYSTEM_REGISTER_TRAP


class AdvSimdFpAccessTrapException(ArmV8Exception):
    """
    A SIMD or floating point instruction was executed while CPACR_EL1.FPEN disallows it.

    CPACR_EL1 resets to trapping everything, so bare-metal startup code must enable FP
    before any vector instruction runs. Firmware that skips that step faults here, which
    is what real hardware does too.
    """
    exception_type = ExceptionType.ADVSIMD_FP_ACCESS_TRAP


class IRQException(ArmV8Exception):
    """
    Asynchronous physical IRQ. Delivered by the board layer rather than raised by an
    instruction, so it carries no syndrome information.
    """
    exception_type = ExceptionType.IRQ


class FIQException(ArmV8Exception):
    exception_type = ExceptionType.FIQ


class SErrorException(ArmV8Exception):
    exception_type = ExceptionType.SERROR
