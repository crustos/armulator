from armulator.armv8.enums import InstrSet
from armulator.armv8.opcodes.decoders import a64_instruction_set


def decode_instruction(instr, processor):
    """
    Route to the decoder for the current execution state. The Cortex-A57 supports AArch32
    at EL0, but that path is not modelled yet - the ARMv6 package remains the AArch32
    implementation for now.
    """
    if processor.registers.current_instr_set() == InstrSet.A64:
        return a64_instruction_set.decode_instruction(instr, processor)
    return None
