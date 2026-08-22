from enum import Enum, auto

__all__ = ['EL', 'InstrSet', 'BranchType', 'ExceptionType', 'Constraint', 'MemOp', 'MemAtomicOp',
           'AccType', 'MBReqDomain', 'MBReqTypes', 'SystemHintOp', 'PSTATEField', 'CountOp',
           'ExtendType', 'ShiftType', 'LogicalOp', 'MoveWideOp', 'RevOp', 'CompareOp', 'Unpredictable']


class EL(Enum):
    """
    Exception levels. Values match PSTATE.EL encoding.
    """
    EL0 = 0b00
    EL1 = 0b01
    EL2 = 0b10
    EL3 = 0b11


class InstrSet(Enum):
    """
    On the Cortex-A57 only A64, A32 and T32 are implemented (no Jazelle/ThumbEE).
    """
    A64 = auto()
    A32 = auto()
    T32 = auto()


class BranchType(Enum):
    CALL = auto()
    ERET = auto()
    DBGEXIT = auto()
    RET = auto()
    JMP = auto()
    EXCEPTION = auto()
    UNKNOWN = auto()


class ExceptionType(Enum):
    UNCATEGORIZED = auto()
    WFX_TRAP = auto()
    CP15RT_TRAP = auto()
    CP15RRT_TRAP = auto()
    CP14RT_TRAP = auto()
    CP14DT_TRAP = auto()
    ADVSIMD_FP_ACCESS_TRAP = auto()
    FP_ID_TRAP = auto()
    CP14RRT_TRAP = auto()
    ILLEGAL_STATE = auto()
    SUPERVISOR_CALL = auto()
    HYPERVISOR_CALL = auto()
    MONITOR_CALL = auto()
    SYSTEM_REGISTER_TRAP = auto()
    INSTRUCTION_ABORT = auto()
    PC_ALIGNMENT = auto()
    DATA_ABORT = auto()
    SP_ALIGNMENT = auto()
    FP_TRAPPED_EXCEPTION = auto()
    SERROR = auto()
    BREAKPOINT = auto()
    SOFTWARE_STEP = auto()
    WATCHPOINT = auto()
    SOFTWARE_BREAKPOINT = auto()
    VECTOR_CATCH = auto()
    IRQ = auto()
    FIQ = auto()


class Constraint(Enum):
    """
    CONSTRAINED UNPREDICTABLE resolution choices.
    """
    NONE = auto()
    UNKNOWN = auto()
    UNDEF = auto()
    NOP = auto()
    WBSUPPRESS = auto()
    FAULT = auto()


class MemOp(Enum):
    LOAD = auto()
    STORE = auto()
    PREFETCH = auto()


class MemAtomicOp(Enum):
    ADD = auto()
    BIC = auto()
    EOR = auto()
    ORR = auto()
    SMAX = auto()
    SMIN = auto()
    UMAX = auto()
    UMIN = auto()
    SWP = auto()


class AccType(Enum):
    NORMAL = auto()
    UNPRIV = auto()
    VEC = auto()
    VECSTREAM = auto()
    ATOMIC = auto()
    ORDERED = auto()
    ORDEREDATOMIC = auto()
    LIMITEDORDERED = auto()
    IFETCH = auto()
    PTW = auto()
    DC = auto()
    IC = auto()
    AT = auto()


class MBReqDomain(Enum):
    NONSHAREABLE = auto()
    INNER_SHAREABLE = auto()
    OUTER_SHAREABLE = auto()
    FULL_SYSTEM = auto()


class MBReqTypes(Enum):
    READS = auto()
    WRITES = auto()
    ALL = auto()


class SystemHintOp(Enum):
    NOP = auto()
    YIELD = auto()
    WFE = auto()
    WFI = auto()
    SEV = auto()
    SEVL = auto()


class PSTATEField(Enum):
    DAIFSET = auto()
    DAIFCLR = auto()
    SP = auto()


class CountOp(Enum):
    CLZ = auto()
    CLS = auto()


class ExtendType(Enum):
    UXTB = 0b000
    UXTH = 0b001
    UXTW = 0b010
    UXTX = 0b011
    SXTB = 0b100
    SXTH = 0b101
    SXTW = 0b110
    SXTX = 0b111


class ShiftType(Enum):
    LSL = 0b00
    LSR = 0b01
    ASR = 0b10
    ROR = 0b11


class LogicalOp(Enum):
    AND = auto()
    EOR = auto()
    ORR = auto()


class MoveWideOp(Enum):
    N = auto()
    Z = auto()
    K = auto()


class RevOp(Enum):
    REV16 = auto()
    REV32 = auto()
    REV64 = auto()


class CompareOp(Enum):
    GT = auto()
    GE = auto()
    EQ = auto()
    LE = auto()
    LT = auto()


class Unpredictable(Enum):
    WBOVERLAPLD = auto()
    WBOVERLAPST = auto()
    LDPOVERLAP = auto()
    BASEOVERLAP = auto()
    DATAOVERLAP = auto()
