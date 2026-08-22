from armulator.armv8.opcodes.opcode import Opcode


class SystemInstruction(Opcode):
    """
    SYS and SYSL: the cache, TLB and address translation maintenance operations
    (DC, IC, AT, TLBI).

    There are no caches to maintain, so DC and IC retire without effect. TLBI does have
    something to do: it drops the emulator's cached translations, so firmware that edits
    a page table and then invalidates sees its change take effect.
    """

    def __init__(self, instruction, t, op1, crn, crm, op2, read):
        super().__init__(instruction)
        self.t = t
        self.op1 = op1
        self.crn = crn
        self.crm = crm
        self.op2 = op2
        self.read = read

    #: CRn values 8 and 9 are the TLB maintenance space.
    TLBI_CRN = (0b1000, 0b1001)

    def execute(self, processor):
        if self.crn in self.TLBI_CRN and not self.read:
            # Any TLBI is treated as invalidate-all. Narrower forms exist (by ASID, by
            # address) but over-invalidating is always architecturally safe.
            processor.mmu.flush()
        if self.read:
            # SYSL reads back an IMPLEMENTATION DEFINED value; zero is a valid choice.
            processor.registers.set_x(self.t, 0)
