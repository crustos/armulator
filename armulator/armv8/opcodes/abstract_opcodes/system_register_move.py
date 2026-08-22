from armulator.armv8.opcodes.opcode import Opcode


class SystemRegisterMove(Opcode):
    """
    MRS and MSR (register). The system register is named by the five-field encoding
    (op0, op1, CRn, CRm, op2), which is exactly how the register file is keyed.
    """

    def __init__(self, instruction, t, op0, op1, crn, crm, op2, read):
        super().__init__(instruction)
        self.t = t
        self.op0 = op0
        self.op1 = op1
        self.crn = crn
        self.crm = crm
        self.op2 = op2
        self.read = read

    @property
    def encoding(self):
        return self.op0, self.op1, self.crn, self.crm, self.op2

    #: Writing any of these can change how addresses translate, so cached translations
    #: have to go. Firmware is only required to issue a TLBI after editing a *table*;
    #: repointing TTBR or reconfiguring TCR needs no TLBI, so the flush must happen here.
    TRANSLATION_CONTROL = {
        (0b11, 0b000, 0b0001, 0b0000, 0b000),   # SCTLR_EL1
        (0b11, 0b000, 0b0010, 0b0000, 0b000),   # TTBR0_EL1
        (0b11, 0b000, 0b0010, 0b0000, 0b001),   # TTBR1_EL1
        (0b11, 0b000, 0b0010, 0b0000, 0b010),   # TCR_EL1
        (0b11, 0b000, 0b1010, 0b0010, 0b000),   # MAIR_EL1
    }

    def execute(self, processor):
        if self.read:
            value = processor.registers.get_system_register(*self.encoding)
            processor.registers.set_x(self.t, value)
        else:
            processor.registers.set_system_register(
                *self.encoding, processor.registers.get_x(self.t)
            )
            if self.encoding in self.TRANSLATION_CONTROL:
                processor.mmu.flush()
