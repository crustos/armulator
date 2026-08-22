from armulator.armv8.arm_exceptions import IllegalStateException
from armulator.armv8.enums import EL
from armulator.armv8.opcodes.opcode import Opcode


class ExceptionReturn(Opcode):
    """
    ERET - return from an exception by restoring PSTATE from SPSR_ELx and branching to
    ELR_ELx. Undefined at EL0, which has no saved state to return to.
    """

    def execute(self, processor):
        el = processor.registers.current_el()
        if el == EL.EL0:
            raise IllegalStateException('ERET is not permitted at EL0')
        processor.exception_return(processor.registers.elr[el], processor.registers.spsr[el])
