from armulator.armv8.bits_ops import add_with_carry, bit_not
from armulator.armv8.opcodes.opcode import Opcode


class ConditionalCompare(Opcode):
    """
    CCMP and CCMN. When the condition holds the comparison is performed normally;
    otherwise the flags are simply replaced with the immediate supplied in the
    instruction. That is how a compiler chains several tests together without branching.
    """

    def __init__(self, instruction, n, operand2, condition, flags, sub_op, datasize):
        super().__init__(instruction)
        self.n = n
        self.operand2 = operand2
        self.condition = condition
        self.flags = flags
        self.sub_op = sub_op
        self.datasize = datasize

    def execute(self, processor):
        if processor.registers.condition_holds(self.condition):
            operand1 = processor.registers.get_x(self.n, self.datasize)
            operand2 = self.operand2
            if callable(operand2):
                operand2 = operand2(processor)
            if self.sub_op:
                operand2 = bit_not(operand2, self.datasize)
                carry_in = 1
            else:
                carry_in = 0
            result, carry, overflow = add_with_carry(
                operand1, operand2, carry_in, self.datasize
            )
            processor.registers.set_flags(result, carry, overflow, self.datasize)
        else:
            processor.registers.pstate.nzcv = self.flags
