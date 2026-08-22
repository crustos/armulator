from armulator.armv8.enums import PSTATEField
from armulator.armv8.opcodes.opcode import Opcode


class MsrImmediate(Opcode):
    """
    MSR (immediate) - the small set of PSTATE fields writable with a bare immediate:
    DAIFSet, DAIFClr and SPSel. This is how firmware unmasks interrupts and chooses
    which stack pointer to run on.
    """

    def __init__(self, instruction, field, operand):
        super().__init__(instruction)
        self.field = field
        self.operand = operand

    def execute(self, processor):
        pstate = processor.registers.pstate
        if self.field == PSTATEField.DAIFSET:
            pstate.daif = pstate.daif | self.operand
        elif self.field == PSTATEField.DAIFCLR:
            pstate.daif = pstate.daif & ~self.operand & 0b1111
        elif self.field == PSTATEField.SP:
            pstate.sp = self.operand & 1
