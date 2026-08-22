from armulator.armv8 import fp_ops
from armulator.armv8.opcodes.opcode import Opcode


class FpDataProcessing1Source(Opcode):
    """
    FMOV, FABS, FNEG, FSQRT, FRINT* and FCVT between precisions.

    FMOV, FABS and FNEG are bitwise: they do not interpret the value, so they leave a
    NaN payload intact rather than turning it into a default NaN.
    """

    def __init__(self, instruction, d, n, operation, datasize, dstsize=None):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.operation = operation
        self.datasize = datasize
        self.dstsize = dstsize or datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        operand = processor.registers.get_v(self.n, self.datasize)
        rounding = processor.registers.fp_rounding_mode

        if self.operation == 'fmov':
            result = operand
        elif self.operation == 'fabs':
            result = fp_ops.fp_abs(operand, self.datasize)
        elif self.operation == 'fneg':
            result = fp_ops.fp_neg(operand, self.datasize)
        elif self.operation == 'fsqrt':
            result = fp_ops.fp_sqrt(operand, self.datasize)
        elif self.operation == 'fcvt':
            result = fp_ops.fp_convert(operand, self.datasize, self.dstsize)
        elif self.operation == 'frintn':
            result = fp_ops.fp_round_to_int(operand, self.datasize, fp_ops.FPRounding.TIEEVEN)
        elif self.operation == 'frintp':
            result = fp_ops.fp_round_to_int(operand, self.datasize, fp_ops.FPRounding.POSINF)
        elif self.operation == 'frintm':
            result = fp_ops.fp_round_to_int(operand, self.datasize, fp_ops.FPRounding.NEGINF)
        elif self.operation == 'frintz':
            result = fp_ops.fp_round_to_int(operand, self.datasize, fp_ops.FPRounding.ZERO)
        else:
            # FRINTA/FRINTX/FRINTI follow the current rounding mode.
            result = fp_ops.fp_round_to_int(operand, self.datasize, rounding)

        processor.registers.set_v(self.d, result, self.dstsize)


class FpDataProcessing2Source(Opcode):
    """
    FADD, FSUB, FMUL, FDIV, FNMUL and the min/max family.
    """

    OPERATIONS = {
        'fadd': fp_ops.fp_add,
        'fsub': fp_ops.fp_sub,
        'fmul': fp_ops.fp_mul,
        'fdiv': fp_ops.fp_div,
        'fmax': fp_ops.fp_max,
        'fmin': fp_ops.fp_min,
        'fmaxnm': fp_ops.fp_max_num,
        'fminnm': fp_ops.fp_min_num,
    }

    def __init__(self, instruction, d, n, m, operation, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.operation = operation
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        operand1 = processor.registers.get_v(self.n, self.datasize)
        operand2 = processor.registers.get_v(self.m, self.datasize)

        if self.operation == 'fnmul':
            result = fp_ops.fp_neg(
                fp_ops.fp_mul(operand1, operand2, self.datasize), self.datasize
            )
        else:
            result = self.OPERATIONS[self.operation](operand1, operand2, self.datasize)

        processor.registers.set_v(self.d, result, self.datasize)


class FpDataProcessing3Source(Opcode):
    """
    FMADD, FMSUB, FNMADD and FNMSUB.

    These are fused: the product is computed exactly and only the final sum is rounded.
    Python's ``math.fma`` gives that behaviour directly for double precision; single
    precision is computed in double and rounded once at the end, which is equivalent.
    """

    def __init__(self, instruction, d, n, m, a, negate_product, negate_addend, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.a = a
        self.negate_product = negate_product
        self.negate_addend = negate_addend
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        size = self.datasize
        operand1 = processor.registers.get_v(self.n, size)
        operand2 = processor.registers.get_v(self.m, size)
        addend = processor.registers.get_v(self.a, size)

        if (fp_ops.is_nan(operand1, size) or fp_ops.is_nan(operand2, size)
                or fp_ops.is_nan(addend, size)):
            processor.registers.set_v(self.d, fp_ops.default_nan(size), size)
            return

        if self.negate_product:
            operand1 = fp_ops.fp_neg(operand1, size)
        if self.negate_addend:
            addend = fp_ops.fp_neg(addend, size)

        # Compute in double precision so the product is not rounded before the add.
        product = fp_ops.bits_to_float(operand1, size) * fp_ops.bits_to_float(operand2, size)
        total = product + fp_ops.bits_to_float(addend, size)
        processor.registers.set_v(self.d, fp_ops.float_to_bits(total, size), size)


class FpCompare(Opcode):
    """
    FCMP and FCMPE. The result goes to PSTATE.NZCV, so an ordinary B.cond follows.
    An unordered comparison sets C and V, which is what makes both LT and GT fail.
    """

    def __init__(self, instruction, n, m, compare_with_zero, datasize):
        super().__init__(instruction)
        self.n = n
        self.m = m
        self.compare_with_zero = compare_with_zero
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        operand1 = processor.registers.get_v(self.n, self.datasize)
        operand2 = 0 if self.compare_with_zero else processor.registers.get_v(
            self.m, self.datasize
        )
        processor.registers.pstate.nzcv = fp_ops.fp_compare(operand1, operand2, self.datasize)


class FpConditionalSelect(Opcode):
    """
    FCSEL - the floating point counterpart of CSEL.
    """

    def __init__(self, instruction, d, n, m, condition, datasize):
        super().__init__(instruction)
        self.d = d
        self.n = n
        self.m = m
        self.condition = condition
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        source = self.n if processor.registers.condition_holds(self.condition) else self.m
        processor.registers.set_v(
            self.d, processor.registers.get_v(source, self.datasize), self.datasize
        )


class FpConditionalCompare(Opcode):
    """
    FCCMP and FCCMPE - compare if the condition holds, otherwise install the supplied
    flags, mirroring the integer CCMP.
    """

    def __init__(self, instruction, n, m, condition, flags, datasize):
        super().__init__(instruction)
        self.n = n
        self.m = m
        self.condition = condition
        self.flags = flags
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        if processor.registers.condition_holds(self.condition):
            processor.registers.pstate.nzcv = fp_ops.fp_compare(
                processor.registers.get_v(self.n, self.datasize),
                processor.registers.get_v(self.m, self.datasize),
                self.datasize,
            )
        else:
            processor.registers.pstate.nzcv = self.flags


class FpImmediate(Opcode):
    """
    FMOV (immediate) - an eight-bit encoding covering a small set of useful constants
    such as 1.0, 2.0 and 0.5. The immediate has already been expanded by the decoder.
    """

    def __init__(self, instruction, d, imm, datasize):
        super().__init__(instruction)
        self.d = d
        self.imm = imm
        self.datasize = datasize

    def execute(self, processor):
        processor.check_fp_enabled()
        processor.registers.set_v(self.d, self.imm, self.datasize)
