from armulator.armv8 import fp_ops
from armulator.armv8.opcodes.opcode import Opcode


class FpIntegerConvert(Opcode):
    """
    Movement and conversion between the general purpose and SIMD register files:
    FMOV (general), SCVTF/UCVTF and the FCVT*S/U rounding conversions.

    FMOV reinterprets the bits; the others convert numerically. The distinction matters:
    FMOV of 1 gives a denormal, SCVTF of 1 gives 1.0.
    """

    def __init__(self, instruction, d, n, operation, fp_size, int_size, unsigned,
                 to_general, rounding=fp_ops.FPRounding.ZERO, top_half=False):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.operation = operation
        self.fp_size = fp_size
        self.int_size = int_size
        self.unsigned = unsigned
        self.to_general = to_general
        self.rounding = rounding
        self.top_half = top_half

    def execute(self, processor):
        processor.check_fp_enabled()
        registers = processor.registers

        if self.operation == 'fmov':
            if self.to_general:
                if self.top_half:
                    # FMOV Xd, Vn.D[1] reads the upper half of the vector register.
                    value = registers.get_v_element(self.n, 1, 64)
                else:
                    value = registers.get_v(self.n, self.fp_size)
                registers.set_x(self.d, value, self.int_size)
            else:
                value = registers.get_x(self.n, self.int_size)
                if self.top_half:
                    registers.set_v_element(self.d, 1, 64, value)
                else:
                    registers.set_v(self.d, value, self.fp_size)
            return

        if self.operation == 'cvtf':
            # Integer to floating point.
            value = registers.get_x(self.n, self.int_size)
            registers.set_v(
                self.d,
                fp_ops.fixed_to_fp(value, self.int_size, self.fp_size, self.unsigned),
                self.fp_size,
            )
            return

        # Floating point to integer, with the rounding mode fixed by the mnemonic.
        value = registers.get_v(self.n, self.fp_size)
        registers.set_x(
            self.d,
            fp_ops.fp_to_fixed(value, self.fp_size, self.int_size, self.unsigned,
                               rounding=self.rounding),
            self.int_size,
        )
