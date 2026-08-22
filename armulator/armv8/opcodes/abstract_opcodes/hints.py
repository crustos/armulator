from armulator.armv8.opcodes.opcode import Opcode


class Hint(Opcode):
    """
    NOP, YIELD, WFE, WFI, SEV and SEVL. Unrecognised hints in the reserved space execute
    as NOP, which is what the architecture requires - a hint must never fault.
    """

    def __init__(self, instruction, kind):
        super().__init__(instruction)
        self.kind = kind

    def execute(self, processor):
        if self.kind == 'yield':
            processor.hint_yield()
        elif self.kind == 'wfe':
            if not processor.event_registered():
                processor.wait_for_event()
            else:
                processor.clear_event_register()
        elif self.kind == 'wfi':
            processor.wait_for_interrupt()
        elif self.kind == 'sev':
            processor.send_event()
        elif self.kind == 'sevl':
            processor.send_event_local()
        elif self.kind == 'clrex':
            processor.exclusive_monitor.clear(processor.cpu_id)
        # nop and anything unrecognised fall through and do nothing.
