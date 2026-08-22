from armulator.armv8.arm_exceptions import (
    HVCException,
    SMCException,
    SVCException,
    SoftwareBreakpointException,
)
from armulator.armv8.opcodes.opcode import Opcode


class ExceptionGeneration(Opcode):
    """
    SVC, HVC, SMC, BRK and HLT. Each raises its exception with the 16-bit immediate,
    which the CPU model places in ESR_ELx.ISS for the handler to read.
    """

    EXCEPTIONS = {
        'svc': SVCException,
        'hvc': HVCException,
        'smc': SMCException,
        'brk': SoftwareBreakpointException,
    }

    def __init__(self, instruction, imm16, kind):
        super().__init__(instruction)
        self.imm16 = imm16
        self.kind = kind

    def execute(self, processor):
        if self.kind == 'hlt':
            # HLT halts rather than vectoring; there is no debugger attached to accept it.
            processor.run = False
            return
        raise self.EXCEPTIONS[self.kind](self.imm16)
