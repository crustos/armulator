"""
AArch32 execution state, for running 32-bit code at EL0 under an AArch64 kernel.

The Cortex-A57 can execute AArch32 at EL0, which is how a 64-bit kernel runs 32-bit
applications. The instruction set involved is the same A32/T32 that
:mod:`armulator.armv6` already decodes and executes in full - several hundred opcodes,
all tested. Reimplementing that here would duplicate the lot and then drift out of step
with it.

So instead of a second implementation, this is an adapter. The ARMv6 opcodes are written
against the ARMv6 processor interface - ``processor.registers.get(n)``,
``processor.mem_u_get(...)``, ``processor.condition_passed()`` - so this module presents
that interface, backed by AArch64 state:

===============  ==========================================================
AArch32 view     AArch64 state
===============  ==========================================================
R0-R12           X0-X12, low 32 bits
SP (R13)         X13
LR (R14)         X14
PC (R15)         the program counter
CPSR.NZCV        PSTATE.N/Z/C/V - the same flags, read through another name
CPSR.T           whether T32 or A32 is executing
===============  ==========================================================

The mapping is not a convenience: it is what the architecture specifies. An AArch32
register *is* the low half of the corresponding AArch64 one, which is why a 64-bit kernel
can read a 32-bit application's arguments straight out of X0-X7.

Two things do not carry over. There is no banking by AArch32 processor mode, because at
EL0 there is only one mode - user - and the banked registers of the other modes belong to
exception levels that are executing AArch64. And an exception taken here does not enter an
AArch32 mode at all: it goes to EL1 in AArch64, through the "lower EL using AArch32" group
of the vector table, which is the whole point of the arrangement.
"""

from armulator.armv6.enums import InstrSet as A32InstrSet
from armulator.armv6.opcodes.decode_instruction import decode_instruction as decode_a32
from armulator.armv8.arm_exceptions import (
    DataAbortException,
    HVCException,
    SMCException,
    SVCException,
    SoftwareBreakpointException,
    UndefinedInstructionException,
)
from armulator.armv8.bits_ops import bit_at, chain, lower_chunk, substring

#: AArch32 register numbers whose AArch64 home is not simply Xn.
SP_REGISTER = 13
LR_REGISTER = 14
PC_REGISTER = 15

#: PSTATE.M[3:0] for AArch32 user mode, as it appears in an SPSR.
AARCH32_USER_MODE = 0b0000


class Cpsr:
    """
    The CPSR as AArch32 code sees it, backed by PSTATE.

    The condition flags are not copied: they are the same bits under another name, so a
    flag set by an AArch32 instruction is immediately visible to the AArch64 kernel that
    handles the next exception.
    """

    def __init__(self, view):
        self._view = view

    @property
    def _pstate(self):
        return self._view.processor.registers.pstate

    # -- condition flags, shared with PSTATE --------------------------------
    @property
    def n(self):
        return self._pstate.n

    @n.setter
    def n(self, value):
        self._pstate.n = int(bool(value))

    @property
    def z(self):
        return self._pstate.z

    @z.setter
    def z(self, value):
        self._pstate.z = int(bool(value))

    @property
    def c(self):
        return self._pstate.c

    @c.setter
    def c(self, value):
        self._pstate.c = int(bool(value))

    @property
    def v(self):
        return self._pstate.v

    @v.setter
    def v(self, value):
        self._pstate.v = int(bool(value))

    # -- interrupt masks, also shared ---------------------------------------
    @property
    def i(self):
        return self._pstate.i

    @i.setter
    def i(self, value):
        self._pstate.i = int(bool(value))

    @property
    def f(self):
        return self._pstate.f

    @f.setter
    def f(self, value):
        self._pstate.f = int(bool(value))

    @property
    def a(self):
        return self._pstate.a

    @a.setter
    def a(self, value):
        self._pstate.a = int(bool(value))

    # -- execution state ----------------------------------------------------
    @property
    def t(self):
        return 1 if self._view.instruction_set is A32InstrSet.THUMB else 0

    @t.setter
    def t(self, value):
        self._view.instruction_set = A32InstrSet.THUMB if value else A32InstrSet.ARM

    @property
    def j(self):
        # Jazelle does not exist on an ARMv8 core.
        return 0

    @j.setter
    def j(self, value):
        pass

    @property
    def e(self):
        return 0        # little endian; SCTLR.E0E is not modelled

    @e.setter
    def e(self, value):
        pass

    @property
    def m(self):
        return 0b10000  # user mode, the only AArch32 mode reachable at EL0

    @m.setter
    def m(self, value):
        # AArch32 code at EL0 cannot change mode; the write is ignored rather than
        # faulting, because the instruction that attempts it is UNPREDICTABLE.
        pass

    @property
    def it(self):
        return self._view.it_state

    @it.setter
    def it(self, value):
        self._view.it_state = lower_chunk(value, 8)

    @property
    def value(self):
        """The CPSR assembled into a word, for MRS and for exception entry."""
        result = 0
        result |= self.n << 31
        result |= self.z << 30
        result |= self.c << 29
        result |= self.v << 28
        result |= substring(self.it, 1, 0) << 25
        result |= self.a << 8
        result |= self.i << 7
        result |= self.f << 6
        result |= self.t << 5
        result |= substring(self.it, 7, 2) << 10
        result |= self.m
        return result

    @value.setter
    def value(self, word):
        self.n = bit_at(word, 31)
        self.z = bit_at(word, 30)
        self.c = bit_at(word, 29)
        self.v = bit_at(word, 28)
        self.a = bit_at(word, 8)
        self.i = bit_at(word, 7)
        self.f = bit_at(word, 6)
        self.t = bit_at(word, 5)
        self.it = chain(substring(word, 15, 10), substring(word, 26, 25), 2)


class AArch32Registers:
    """
    The AArch32 register file, as a view onto the AArch64 one.
    """

    def __init__(self, view):
        self._view = view
        self.cpsr = Cpsr(view)
        #: Which registers an instruction wrote, so the PC increment can be suppressed
        #: after a branch. The ARMv6 opcodes rely on this being present and indexable.
        self.changed_registers = [False] * 16

    @property
    def _registers(self):
        return self._view.processor.registers

    def get(self, n):
        if n == PC_REGISTER:
            # Reading the PC in AArch32 yields the address of the current instruction
            # plus eight in A32, or plus four in T32 - the pipeline offset that has
            # tripped up hand-written ARM assembly for thirty years.
            offset = 4 if self._view.instruction_set is A32InstrSet.THUMB else 8
            return lower_chunk(self._view.pc + offset, 32)
        return self._registers.get_x(n, 32)

    def set(self, n, value):
        self.changed_registers[n] = True
        if n == PC_REGISTER:
            self.branch_to(value)
            return
        self._registers.set_x(n, value, 32)

    def get_sp(self):
        return self._registers.get_x(SP_REGISTER, 32)

    def set_sp(self, value):
        self.changed_registers[SP_REGISTER] = True
        self._registers.set_x(SP_REGISTER, value, 32)

    def get_lr(self):
        return self._registers.get_x(LR_REGISTER, 32)

    def set_lr(self, value):
        self.changed_registers[LR_REGISTER] = True
        self._registers.set_x(LR_REGISTER, value, 32)

    def get_pc(self):
        return self.get(PC_REGISTER)

    def pc_store_value(self):
        return self.get(PC_REGISTER)

    def branch_to(self, address):
        self.changed_registers[PC_REGISTER] = True
        self._view.pc = lower_chunk(address, 32)

    def increment_pc(self, length):
        self._view.pc = lower_chunk(self._view.pc + length, 32)

    def current_instr_set(self):
        return self._view.instruction_set

    def select_instr_set(self, instruction_set):
        self._view.instruction_set = instruction_set

    def current_mode_is_not_user(self):
        return False        # EL0 is unprivileged by definition

    def current_mode_is_user_or_system(self):
        return True

    def current_mode_is_hyp(self):
        return False

    def is_secure(self):
        return self._registers.secure

    def it_advance(self):
        """
        Step the IT state machine after a conditional T32 instruction.
        """
        state = self._view.it_state
        if substring(state, 2, 0) == 0b000:
            self._view.it_state = 0
        else:
            self._view.it_state = (substring(state, 7, 5) << 5) | (
                lower_chunk(state << 1, 5))

    def set_event_register(self, flag):
        self._registers.set_event_register(flag)

    def get_event_register(self):
        return self._registers.get_event_register()


class AArch32View:
    """
    Presents the ARMv6 processor interface, backed by an :class:`ArmV8`.

    An instance of this is what the ARMv6 opcodes execute against. Everything they need
    that is genuinely architectural - memory, condition flags, branching - is forwarded to
    the AArch64 core, so both execution states see one machine.
    """

    def __init__(self, processor):
        self.processor = processor
        self.registers = AArch32Registers(self)
        self.instruction_set = A32InstrSet.ARM
        self.it_state = 0
        self.pc = 0
        self.opcode = 0
        self.opcode_len = 0

    # ------------------------------------------------------------------
    # Condition codes
    # ------------------------------------------------------------------

    def current_cond(self):
        if self.instruction_set is A32InstrSet.ARM:
            return substring(self.opcode, 31, 28)
        if self.opcode_len == 16 and substring(self.opcode, 15, 12) == 0b1101:
            return substring(self.opcode, 11, 8)
        if substring(self.it_state, 3, 0) != 0b0000:
            return substring(self.it_state, 7, 4)
        return 0b1110

    def condition_passed(self):
        """
        AArch32's per-instruction condition field, evaluated against the shared flags.
        """
        return self.processor.registers.condition_holds(self.current_cond())

    def in_it_block(self):
        return substring(self.it_state, 3, 0) != 0b0000

    def last_in_it_block(self):
        return substring(self.it_state, 3, 0) == 0b1000

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def branch_write_pc(self, address):
        if self.instruction_set is A32InstrSet.ARM:
            self.registers.branch_to(address & ~0b11)
        else:
            self.registers.branch_to(address & ~0b1)

    def bx_write_pc(self, address):
        """
        Interworking: bit 0 of the target selects T32 or A32, which is how a single
        branch instruction moves between the two instruction sets.
        """
        if bit_at(address, 0):
            self.instruction_set = A32InstrSet.THUMB
            self.registers.branch_to(address & ~0b1)
        else:
            self.instruction_set = A32InstrSet.ARM
            self.registers.branch_to(address & ~0b11)

    def alu_write_pc(self, address):
        self.bx_write_pc(address)

    def load_write_pc(self, address):
        self.bx_write_pc(address)

    # ------------------------------------------------------------------
    # Memory, forwarded to the AArch64 core
    # ------------------------------------------------------------------

    def mem_u_get(self, address, size):
        return self.processor.mem_get(lower_chunk(address, 32), size, aligned=False)

    def mem_u_set(self, address, size, value):
        self.processor.mem_set(lower_chunk(address, 32), size, value, aligned=False)

    def mem_a_get(self, address, size):
        return self.processor.mem_get(lower_chunk(address, 32), size)

    def mem_a_set(self, address, size, value):
        self.processor.mem_set(lower_chunk(address, 32), size, value)

    def mem_u_unpriv_get(self, address, size):
        return self.mem_u_get(address, size)

    def mem_u_unpriv_set(self, address, size, value):
        self.mem_u_set(address, size, value)

    def mem_u_with_priv_get(self, address, size, privileged):
        return self.mem_u_get(address, size)

    def mem_u_with_priv_set(self, address, size, privileged, value):
        self.mem_u_set(address, size, value)

    def big_endian(self):
        return False

    def unaligned_support(self):
        return True

    # ------------------------------------------------------------------
    # Exclusives and events, shared with the AArch64 side
    # ------------------------------------------------------------------

    def exclusive_monitors_pass(self, address, size):
        return self.processor.exclusive_monitor.check_and_clear(
            self.processor.cpu_id, lower_chunk(address, 32))

    def set_exclusive_monitors(self, address, size):
        self.processor.exclusive_monitor.reserve(
            self.processor.cpu_id, lower_chunk(address, 32))

    def clear_exclusive_local(self, processor_id):
        self.processor.exclusive_monitor.clear(self.processor.cpu_id)

    def clear_event_register(self):
        self.processor.clear_event_register()

    def event_registered(self):
        return self.processor.event_registered()

    def send_event(self):
        self.processor.send_event()

    def send_event_local(self):
        self.processor.send_event_local()

    def wait_for_event(self):
        self.processor.wait_for_event()

    def wait_for_interrupt(self):
        self.processor.wait_for_interrupt()

    def hint_yield(self):
        pass

    def hint_preload_data(self, address):
        pass

    def hint_preload_data_for_write(self, address):
        pass

    def hint_preload_instr(self, address):
        pass

    def data_memory_barrier(self, domain, types):
        self.processor.data_memory_barrier(domain, types)

    def data_synchronization_barrier(self, domain, types):
        self.processor.data_synchronization_barrier(domain, types)

    def instruction_synchronization_barrier(self):
        self.processor.instruction_synchronization_barrier()

    def integer_zero_divide_trapping_enabled(self):
        return False

    # ------------------------------------------------------------------
    # Exceptions
    #
    # The ARMv6 opcodes call these to raise an exception. They raise the AArch64
    # exception classes, so entry goes through the same routing as anything else and
    # lands at EL1 in AArch64 rather than in an AArch32 mode.
    # ------------------------------------------------------------------

    def call_supervisor(self, immediate):
        raise SVCException(lower_chunk(immediate, 16))

    def call_hypervisor(self, immediate):
        raise HVCException(lower_chunk(immediate, 16))

    def call_secure_monitor(self, immediate):
        raise SMCException(lower_chunk(immediate, 16))

    def bkpt_instr_debug_event(self):
        raise SoftwareBreakpointException(0)

    def generate_coprocessor_exception(self):
        raise UndefinedInstructionException('coprocessor access from AArch32 EL0')

    def generate_integer_zero_divide(self):
        raise UndefinedInstructionException('integer divide by zero')

    def coproc_accepted(self, cp_num, instr):
        # Coprocessor access from AArch32 EL0 is not modelled; VFP and NEON would be
        # reached this way, and they use the AArch64 register file underneath.
        raise UndefinedInstructionException(f'coprocessor {cp_num} not modelled')

    def alignment_fault(self, address, is_write):
        raise DataAbortException(lower_chunk(address, 32), is_write,
                                 'unaligned AArch32 access', status=0b100001)

    def take_hyp_trap_exception(self):
        raise HVCException(0)

    def switch_to_jazelle_execution(self):
        raise UndefinedInstructionException('Jazelle is not implemented on ARMv8')

    def this_instr(self):
        return self.opcode

    def this_instr_length(self):
        return self.opcode_len

    def null_check_if_thumbee(self, n):
        pass

    # ------------------------------------------------------------------
    # Fetch and decode
    # ------------------------------------------------------------------

    def fetch_instruction(self):
        """
        Read the next A32 or T32 instruction. T32 is variable length: a halfword whose
        top bits mark it as the first of a pair is joined with the one after it.
        """
        address = lower_chunk(self.pc, 32)
        if self.instruction_set is A32InstrSet.ARM:
            self.opcode_len = 32
            self.opcode = self.processor.mem_get(address, 4)
            return self.opcode

        self.opcode = self.processor.mem_get(address, 2)
        self.opcode_len = 16
        if substring(self.opcode, 15, 11) in (0b11101, 0b11110, 0b11111):
            second = self.processor.mem_get(lower_chunk(address + 2, 32), 2)
            self.opcode = chain(self.opcode, second, 16)
            self.opcode_len = 32
        return self.opcode

    def decode_instruction(self, instruction):
        return decode_a32(instruction, self)

    def instruction_bytes(self):
        return self.opcode_len // 8
