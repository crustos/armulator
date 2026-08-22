"""
AArch64 processor model, configured for the Cortex-A57 found on the Jetson Nano.

Milestone scope: EL0/EL1 with the MMU off, so address translation is flat. The exception
model is built out properly from the start because the vector layout and ESR encoding are
structural - they are hard to retrofit once instructions depend on them.
"""

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.armv6.memory_attributes import MemoryAttributes, MemType
from armulator.armv6.memory_controller_hub import MemoryControllerHub
from armulator.armv6 import arm_exceptions as a32_exceptions
from armulator.armv8.arm_exceptions import (
    AdvSimdFpAccessTrapException,
    ArmV8Exception,
    DataAbortException,
    EndOfInstruction,
    FIQException,
    HVCException,
    IRQException,
    IllegalStateException,
    InstructionAbortException,
    PcAlignmentException,
    SMCException,
    SVCException,
    SoftwareBreakpointException,
    SpAlignmentException,
    SystemRegisterTrapException,
    UndefinedInstructionException,
)
from armulator.armv8.bits_ops import (
    align,
    sign_extend,
    big_endian_reverse,
    bit_at,
    lower_chunk,
    set_substring,
    substring,
)
from armulator.armv8.enums import EL, ExceptionType
from armulator.armv8.exclusive_monitor import ExclusiveMonitor
from armulator.armv8.memory_history import Clock, MemoryHistory
from armulator.armv8.store_buffer import LoadReorderer, MemoryModel, StoreBuffer
from armulator.armv8.mmu import Mmu, TranslationFault
from armulator.armv8.opcodes.decode_instruction import decode_instruction as op_decode_instruction
from armulator.armv8.registers import Registers

# Offsets into the vector table at VBAR_ELx, in the architectural order.
VECTOR_OFFSET_CURRENT_EL_SP0 = 0x000
VECTOR_OFFSET_CURRENT_EL_SPX = 0x200
VECTOR_OFFSET_LOWER_EL_A64 = 0x400
VECTOR_OFFSET_LOWER_EL_A32 = 0x600

# Offsets within each of the four groups above.
VECTOR_OFFSET_SYNC = 0x000
VECTOR_OFFSET_IRQ = 0x080
VECTOR_OFFSET_FIQ = 0x100
VECTOR_OFFSET_SERROR = 0x180

# Exception class values for ESR_ELx.EC.
EC_UNKNOWN = 0b000000
EC_ILLEGAL_STATE = 0b001110
EC_SVC_A32 = 0b010001
EC_HVC_A32 = 0b010010
EC_SMC_A32 = 0b010011
EC_SVC_A64 = 0b010101
EC_HVC_A64 = 0b010110
EC_SMC_A64 = 0b010111
EC_ADVSIMD_FP_ACCESS = 0b000111
EC_SYSTEM_REGISTER = 0b011000
EC_INSTRUCTION_ABORT_LOWER = 0b100000
EC_INSTRUCTION_ABORT_CURRENT = 0b100001
EC_PC_ALIGNMENT = 0b100010
EC_DATA_ABORT_LOWER = 0b100100
EC_DATA_ABORT_CURRENT = 0b100101
EC_SP_ALIGNMENT = 0b100110
EC_BRK_A64 = 0b111100


class ArmV8:
    #: Highest exception level this model implements. A Cortex-A57 has all four.
    HIGHEST_EL = EL.EL3

    def __init__(self, memory_list=None, cpu_id=0, exclusive_monitor=None,
                 memory_history=None, clock=None, highest_el=None):
        #: Index of this core within its cluster. Reported through MPIDR_EL1 and used to
        #: tell reservations and interrupt targets apart.
        self.cpu_id = cpu_id
        #: The highest exception level available. Lowering it models a part with no
        #: EL3, and makes routing fall back to whatever does exist.
        self.highest_el = highest_el or self.HIGHEST_EL
        #: Shared with the rest of the cluster; a lone core still gets one so that the
        #: exclusive instructions take the same path either way.
        self.exclusive_monitor = exclusive_monitor or ExclusiveMonitor()
        self.registers = Registers()
        self.run = True
        self.opcode = 0
        self.opcode_len = 32
        self.mem = MemoryControllerHub.from_memory_list(memory_list or [])
        self.is_wait_for_event = False
        self.is_wait_for_interrupt = False
        self.executed_opcode = None
        #: Stage 1 address translation for the EL1&0 regime.
        self.mmu = Mmu(self)
        #: Instructions retired by this core, used to age entries in the store buffer.
        self.instruction_count = 0
        #: Stores issued by this core. A deferred load only moves if one of these
        #: happened after it - see note_deferred_load.
        self.store_count = 0
        #: Set when SEV executes, collected by the cluster scheduler.
        self.event_signalled = False
        #: Set by a Cluster so SMC can be serviced as a PSCI call.
        self.psci_handler = None
        #: AArch32 execution state, used when PSTATE.nRW selects it. Built lazily,
        #: because most cores never leave AArch64.
        self._aarch32 = None
        #: Pending stores not yet visible to other cores. Sequential by default, so
        #: behaviour only changes when a relaxed model is asked for explicitly.
        self.store_buffer = StoreBuffer(model=MemoryModel.SEQUENTIAL)
        #: Shared with the cluster: one tick, so events on different cores are ordered
        #: against each other rather than against per-core instruction counts.
        self.clock = clock or Clock()
        #: What memory used to contain, so a reordered load can be answered.
        self.memory_history = memory_history or MemoryHistory()
        #: Decides how far back each load may look.
        self.load_reorderer = LoadReorderer(model=MemoryModel.SEQUENTIAL)
        #: Global time of this core's most recent write to each byte. A load must never
        #: return something older than the core's own store, whatever the reordering
        #: rule says, or single-threaded code would break.
        self.own_write_time = {}
        #: Number of exceptions taken since reset. Lets a scheduler tell a firmware that
        #: has parked on a halt loop from one that is faulting round the vector table,
        #: which otherwise look identical from the outside - the PC repeats either way.
        self.exception_count = 0

    def start(self):
        self.take_reset()

    @property
    def aarch32(self):
        """
        The AArch32 view of this core, creating it on first use.
        """
        if self._aarch32 is None:
            from armulator.armv8.aarch32 import AArch32View
            self._aarch32 = AArch32View(self)
        return self._aarch32

    def enter_aarch32(self, address, thumb=False):
        """
        Switch to AArch32 at EL0 and begin executing there.

        This is what an AArch64 kernel does when it schedules a 32-bit application: set
        PSTATE.nRW, point the AArch32 PC at the entry point, and run. Returning happens
        through an exception, which re-enters AArch64 at EL1.
        """
        from armulator.armv6.enums import InstrSet as A32InstrSet
        self.registers.pstate.n_rw = 1
        self.registers.pstate.el = EL.EL0
        # AArch32 user mode is M[4:0] = 10000, so the low bit of the mode field - which
        # PSTATE.SP occupies in AArch64 - is clear.
        self.registers.pstate.sp = 0
        view = self.aarch32
        view.instruction_set = A32InstrSet.THUMB if thumb else A32InstrSet.ARM
        view.it_state = 0
        # A caller may pass an interworking address with bit 0 set to mean Thumb; the
        # PC itself is always aligned to the instruction size.
        view.pc = lower_chunk(address & ~0b1 if thumb else address & ~0b11, 32)
        self.registers.branch_taken = False

    def emulate_aarch32_cycle(self):
        """
        Execute one AArch32 instruction.

        Decoding and execution are the ARMv6 core's, unchanged: the opcodes run against
        the AArch32 view, which is backed by this processor's registers and memory.
        """
        view = self.aarch32
        preferred_return = view.pc
        try:
            instruction = view.fetch_instruction()
            opcode_class = view.decode_instruction(instruction)
            if not opcode_class:
                raise UndefinedInstructionException(
                    f'undefined AArch32 instruction 0x{instruction:08X}')
            opcode = opcode_class.from_bitarray(instruction, view)
            if opcode is None:
                raise UndefinedInstructionException(
                    f'undefined AArch32 instruction 0x{instruction:08X}')
            view.registers.changed_registers = [False] * 16
            if view.in_it_block():
                opcode.execute(view)
                view.registers.it_advance()
            else:
                opcode.execute(view)
            if not view.registers.changed_registers[15]:
                view.registers.increment_pc(view.instruction_bytes())
        except EndOfInstruction:
            view.registers.increment_pc(view.instruction_bytes())
        except a32_exceptions.EndOfInstruction:
            view.registers.increment_pc(view.instruction_bytes())
        except (SVCException, HVCException, SMCException,
                SoftwareBreakpointException) as exception:
            self._take_aarch32_exception(
                exception, lower_chunk(preferred_return + view.instruction_bytes(), 32))
        except ArmV8Exception as exception:
            self._take_aarch32_exception(exception, preferred_return)
        except a32_exceptions.ArmulatorException as exception:
            # The ARMv6 opcodes raise their own exception classes. Translating them here
            # rather than editing several hundred opcodes keeps the two cores' semantics
            # in one place, and means AArch32 exceptions route through exactly the same
            # AArch64 machinery as everything else.
            self._take_aarch32_exception(
                self._translate_a32_exception(exception), preferred_return)

    #: ARMv6 exception classes and the AArch64 exception each becomes.
    A32_EXCEPTION_MAP = {
        a32_exceptions.UndefinedInstructionException: UndefinedInstructionException,
        a32_exceptions.SVCException: SVCException,
        a32_exceptions.SMCException: SMCException,
        a32_exceptions.HypTrapException: HVCException,
    }

    @staticmethod
    def _describe_a32_exception(exception):
        """
        Describe an ARMv6 exception without calling ``str`` on it.

        Some of the ARMv6 exception classes format themselves from ``args[0]`` and are
        raised with no arguments, so stringifying one raises IndexError - which would
        replace a clean architectural fault with a crash in the handler.
        """
        return f'{type(exception).__name__} from AArch32'

    def _translate_a32_exception(self, exception):
        description = self._describe_a32_exception(exception)
        if isinstance(exception, a32_exceptions.DataAbortException):
            return DataAbortException(
                self.registers.far.get(EL.EL1, 0), False, description)
        for a32_class, a64_class in self.A32_EXCEPTION_MAP.items():
            if isinstance(exception, a32_class):
                if a64_class in (SVCException, SMCException, HVCException):
                    return a64_class(0, description)
                return a64_class(description)
        return UndefinedInstructionException(description)

    def _take_aarch32_exception(self, exception, preferred_return):
        """
        Take an exception raised while executing AArch32.

        Entry is into AArch64 - there are no AArch32 exception levels here - through the
        "lower EL using AArch32" group of the vector table. The kernel that handles it
        sees a 64-bit machine, with the 32-bit register values sitting in the low halves
        of X0-X14 where it left them.
        """
        # The PC lives in the AArch32 view while executing there, so it has to be put
        # back where take_exception expects to find it.
        self.registers.branch_to(preferred_return)
        self.registers.branch_taken = False
        self.take_exception(exception, None, preferred_return)

    def take_reset(self):
        """
        Reset into AArch64 EL1 using SP_EL1, with all exceptions masked.
        """
        self.registers.reset()
        self.registers.set_mpidr(self.cpu_id)
        self.exclusive_monitor.clear(self.cpu_id)
        self.mmu.flush()
        self.is_wait_for_event = False
        self.is_wait_for_interrupt = False
        self.exception_count = 0
        self.run = True

    def print_registers(self):
        print(self.registers.format_registers())

    # ------------------------------------------------------------------
    # Address translation
    # ------------------------------------------------------------------

    def translate_address(self, address, acc_type=None, is_write=False, size=1,
                          was_aligned=True, is_instruction=False):
        """
        Translate a virtual address.

        With SCTLR_EL1.M clear the VA is the PA and every region is Normal memory. With
        it set the stage 1 tables are walked, and a failure becomes an instruction or
        data abort carrying the architectural fault status.
        """
        if self.registers.mmu_enabled:
            try:
                return self.mmu.translate(address, is_write=is_write,
                                          is_instruction=is_instruction)
            except TranslationFault as fault:
                self._record_fault_address(fault)
                if is_instruction:
                    abort = InstructionAbortException(
                        fault.address, str(fault), status=fault.status
                    )
                else:
                    abort = DataAbortException(
                        fault.address, is_write, str(fault), status=fault.status
                    )
                # A stage 2 fault is routed to EL2 regardless of where it happened,
                # because the guest cannot see the tables that produced it.
                abort.from_stage2 = fault.stage2
                raise abort from None

        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = lower_chunk(address, 48)
        descriptor.paddress.ns = 0 if self.registers.current_el() == EL.EL3 else 1
        descriptor.memattrs = MemoryAttributes()
        descriptor.memattrs.type = MemType.NORMAL
        descriptor.memattrs.shareable = False
        descriptor.memattrs.outershareable = False
        return descriptor

    def _record_fault_address(self, fault):
        """
        Put the faulting address where the level that will handle it looks for it.
        """
        level = EL.EL2 if fault.stage2 else self.registers.regime()
        if level not in self.registers.far:
            level = EL.EL1
        self.registers.far[level] = lower_chunk(fault.address, 64)
        if fault.stage2 and fault.intermediate_address is not None:
            # HPFAR_EL2 carries the intermediate address, shifted down by eight, which
            # is what tells a hypervisor which guest page was being reached for.
            self.registers.set_system_register(
                0b11, 0b100, 0b0110, 0b0000, 0b100,
                (fault.intermediate_address >> 8) & ((1 << 40) - 1),
            )

    def big_endian(self):
        """
        SCTLR_EL1.EE selects the data endianness for EL1, E0E for EL0.
        """
        sctlr = self.registers.sctlr_el1
        if self.registers.current_el() == EL.EL0:
            return bool(substring(sctlr, 24, 24))
        return bool(substring(sctlr, 25, 25))

    # ------------------------------------------------------------------
    # Memory accessors
    # ------------------------------------------------------------------

    def mem_get(self, address, size, acc_type=None, aligned=True):
        """
        Mem[address, size] - read. Accesses wider than 8 bytes are split, since the
        memory hub works in 1/2/4/8 byte units.
        """
        address = lower_chunk(address, 64)
        if aligned and self.registers.alignment_check and address != align(address, size):
            self.registers.far[EL.EL1] = address
            raise DataAbortException(address, False, 'unaligned access with SCTLR_EL1.A set')
        if size in (1, 2, 4, 8):
            descriptor = self.translate_address(address, acc_type, False, size, aligned)
            value = self.mem[descriptor, size]
            value = self._apply_load_reordering(descriptor, address, size, value)
            value = self._apply_store_forwarding(address, size, value)
            if self.big_endian():
                value = big_endian_reverse(value, size)
            return value
        value = 0
        for i in range(size):
            byte = self.mem_get(address + i, 1, acc_type, False)
            value = set_substring(value, 8 * i + 7, 8 * i, byte)
        return value

    def mem_set(self, address, size, value, acc_type=None, aligned=True):
        """
        Mem[address, size] = value - write.
        """
        address = lower_chunk(address, 64)
        # A deferred load that overlaps this store has to happen first.
        self.registers.resolve_pending_loads_overlapping(address, size)
        self.store_count += 1
        if aligned and self.registers.alignment_check and address != align(address, size):
            self.registers.far[EL.EL1] = address
            raise DataAbortException(address, True, 'unaligned access with SCTLR_EL1.A set')
        if size in (1, 2, 4, 8):
            descriptor = self.translate_address(address, acc_type, True, size, aligned)
            if self.big_endian():
                value = big_endian_reverse(value, size)
            value = lower_chunk(value, size * 8)
            # A plain store has to break other cores' reservations, otherwise a lock
            # released with STR would leave a stale reservation looking valid. This
            # happens when the store issues, not when it drains.
            self.exclusive_monitor.notify_store(self.cpu_id, address, size)
            if self._buffer_store(descriptor, address, size, value):
                return
            self._commit_store(descriptor, address, size, value)
            return
        for i in range(size):
            self.mem_set(address + i, 1, substring(value, 8 * i + 7, 8 * i), acc_type, False)

    def check_fp_enabled(self):
        """
        AArch64.CheckFPAdvSIMDEnabled - every SIMD and floating point instruction calls
        this before touching a V register.
        """
        if not self.registers.fp_access_enabled:
            raise AdvSimdFpAccessTrapException(
                'SIMD/FP access trapped: set CPACR_EL1.FPEN before using vector registers'
            )

    def _is_bufferable(self, descriptor):
        """
        Only Normal memory may be buffered. Device accesses have side effects at the
        peripheral, so they must reach it in program order and on time - buffering a
        write to a UART would delay the character, and buffering a write to a GPIO
        would let a later read see the pin unchanged.
        """
        return descriptor.memattrs.type == MemType.NORMAL

    def _buffer_store(self, descriptor, address, size, value):
        """
        Try to buffer a store. Returns True when it was taken by the buffer.
        """
        if not self.store_buffer.buffering or not self._is_bufferable(descriptor):
            return False
        if not self.store_buffer.push(address, size, value, self.instruction_count):
            # Buffer over capacity: drain it, as a real one would when it fills.
            self.drain_store_buffer()
        return True

    def _apply_load_reordering(self, descriptor, address, size, value):
        """
        Answer a load as of an earlier time when the reordering rule permits it.

        Device memory is never reordered - a peripheral read has to see the peripheral
        as it is now - and a byte this core wrote since the effective read time is always
        returned at its current value, so a core still observes its own writes in order.
        """
        if not self.load_reorderer.reordering or not self._is_bufferable(descriptor):
            return value
        when = self.load_reorderer.read_time(address, self.clock.now)
        if when >= self.clock.now:
            return value

        stale = self.memory_history.value_as_of(address, size, when, value)
        # Overlay anything this core wrote more recently than the effective read time.
        for offset in range(size):
            byte_address = address + offset
            if self.own_write_time.get(byte_address, -1) > when:
                shift = 8 * offset
                stale = (stale & ~(0xFF << shift)) | (value & (0xFF << shift))
        return stale

    def _apply_store_forwarding(self, address, size, value):
        """
        Overlay this core's own pending stores onto a value read from memory.

        A core must always observe its own writes in program order, so store forwarding
        is what keeps single-threaded code correct while the buffer is in use.
        """
        if not self.store_buffer.entries:
            return value
        available = self.store_buffer.forward(address, size)
        if not available:
            return value
        for byte_address, byte in available.items():
            shift = 8 * (byte_address - address)
            value = (value & ~(0xFF << shift)) | (byte << shift)
        return value

    def retire_pending_stores(self):
        """
        Let the store buffer make progress on its own, as hardware does between
        instructions rather than only when told to.
        """
        def write(address, size, value):
            descriptor = self.translate_address(address, None, True, size, True)
            self._commit_store(descriptor, address, size, value)

        return self.store_buffer.retire(self.instruction_count, write)

    def _commit_store(self, descriptor, address, size, value):
        """
        Push a store out to memory and note when it became globally visible.
        """
        # Read what was there first, so the history knows what this write replaced.
        previous = self.mem[descriptor, size] if self.load_reorderer.reordering else None
        self.mem[descriptor, size] = value
        now = self.clock.now
        self.memory_history.record(address, size, value, now, previous)
        for offset in range(size):
            self.own_write_time[address + offset] = now

    def drain_store_buffer(self):
        """
        Make every pending store visible to the rest of the cluster.
        """
        def write(address, size, value):
            descriptor = self.translate_address(address, None, True, size, True)
            self._commit_store(descriptor, address, size, value)

        return self.store_buffer.drain(write)

    def set_memory_model(self, model):
        """Switch this core's memory model, draining anything already pending."""
        self.drain_store_buffer()
        self.store_buffer.model = model
        self.load_reorderer.model = model
        self.synchronize_reads()

    def check_sp_alignment(self, address=None):
        """
        SP must be 16-byte aligned when used as a base address and SCTLR_EL1.SA is set.
        """
        sp = self.registers.get_sp() if address is None else address
        if self.registers.stack_alignment_check and sp != align(sp, 16):
            raise SpAlignmentException('SP is not 16 byte aligned')

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def branch_to(self, address, branch_type=None):
        self.registers.branch_to(address)

    def branch_write_pc(self, address, branch_type=None):
        self.branch_to(lower_chunk(address, 64), branch_type)

    # ------------------------------------------------------------------
    # Exception entry
    # ------------------------------------------------------------------

    def exception_vector_offset(self, target_el, exception_type):
        """
        Pick the vector table entry: which group depends on where the exception came
        from, which slot within the group depends on the kind of exception.
        """
        source_el = self.registers.current_el()
        if source_el == target_el:
            # PSTATE.SP selects between the SP_EL0 and SP_ELx groups.
            group = VECTOR_OFFSET_CURRENT_EL_SPX if self.registers.pstate.sp else VECTOR_OFFSET_CURRENT_EL_SP0
        elif self.registers.using_aarch32():
            group = VECTOR_OFFSET_LOWER_EL_A32
        else:
            group = VECTOR_OFFSET_LOWER_EL_A64
        if exception_type == ExceptionType.IRQ:
            slot = VECTOR_OFFSET_IRQ
        elif exception_type == ExceptionType.FIQ:
            slot = VECTOR_OFFSET_FIQ
        elif exception_type == ExceptionType.SERROR:
            slot = VECTOR_OFFSET_SERROR
        else:
            slot = VECTOR_OFFSET_SYNC
        return group + slot

    def exception_syndrome(self, exception, target_el):
        """
        Build ESR_ELx for the exception being taken.
        """
        source_el = self.registers.current_el()
        from_lower = source_el != target_el
        exception_type = exception.exception_type
        iss = 0
        il = 1  # A64 instructions are always 32-bit, so IL is set.

        # A call made from AArch32 has its own exception class, so the handler can tell
        # which execution state its caller was in without inspecting the SPSR.
        from_aarch32 = self.registers.using_aarch32()

        if exception_type == ExceptionType.SUPERVISOR_CALL:
            ec = EC_SVC_A32 if from_aarch32 else EC_SVC_A64
            iss = lower_chunk(exception.immediate, 16)
        elif exception_type == ExceptionType.HYPERVISOR_CALL:
            ec = EC_HVC_A32 if from_aarch32 else EC_HVC_A64
            iss = lower_chunk(exception.immediate, 16)
        elif exception_type == ExceptionType.MONITOR_CALL:
            ec = EC_SMC_A32 if from_aarch32 else EC_SMC_A64
            iss = lower_chunk(exception.immediate, 16)
        elif exception_type == ExceptionType.SOFTWARE_BREAKPOINT:
            ec = EC_BRK_A64
            iss = lower_chunk(exception.immediate, 16)
        elif exception_type == ExceptionType.SYSTEM_REGISTER_TRAP:
            ec = EC_SYSTEM_REGISTER
        elif exception_type == ExceptionType.ADVSIMD_FP_ACCESS_TRAP:
            ec = EC_ADVSIMD_FP_ACCESS
        elif exception_type == ExceptionType.INSTRUCTION_ABORT:
            ec = EC_INSTRUCTION_ABORT_LOWER if from_lower else EC_INSTRUCTION_ABORT_CURRENT
            iss = set_substring(iss, 5, 0, getattr(exception, 'status', 0))
        elif exception_type == ExceptionType.DATA_ABORT:
            ec = EC_DATA_ABORT_LOWER if from_lower else EC_DATA_ABORT_CURRENT
            # ISS.WnR reports whether the faulting access was a write, and the low bits
            # carry the fault status so a handler can tell a permission fault from a
            # missing translation.
            iss = set_substring(iss, 6, 6, 1 if getattr(exception, 'is_write', False) else 0)
            iss = set_substring(iss, 5, 0, getattr(exception, 'status', 0))
        elif exception_type == ExceptionType.PC_ALIGNMENT:
            ec = EC_PC_ALIGNMENT
        elif exception_type == ExceptionType.SP_ALIGNMENT:
            ec = EC_SP_ALIGNMENT
        elif exception_type == ExceptionType.ILLEGAL_STATE:
            ec = EC_ILLEGAL_STATE
        else:
            ec = EC_UNKNOWN
            il = 1

        esr = 0
        esr = set_substring(esr, 31, 26, ec)
        esr = set_substring(esr, 25, 25, il)
        esr = set_substring(esr, 24, 0, iss)
        return esr

    def target_el_for(self, exception):
        """
        Decide which exception level takes ``exception``.

        Routing is the heart of the exception level model. A hypervisor claims a guest's
        interrupts by setting HCR_EL2.IMO; secure firmware claims them with SCR_EL3.IRQ;
        a call instruction goes to the level it names. Two rules apply throughout:
        an exception never targets a level below the one it came from, and it never
        targets a level that is not implemented.
        """
        registers = self.registers
        source = registers.current_el()
        kind = exception.exception_type

        if kind == ExceptionType.MONITOR_CALL:
            target = EL.EL3
        elif kind == ExceptionType.HYPERVISOR_CALL:
            target = EL.EL3 if source == EL.EL3 else EL.EL2
        elif kind in (ExceptionType.IRQ, ExceptionType.FIQ, ExceptionType.SERROR):
            target = self._async_target_el(kind, source)
        elif kind == ExceptionType.SUPERVISOR_CALL:
            # HCR_EL2.TGE routes what would have been EL1's exceptions to EL2, which is
            # how a hypervisor runs an application without a guest kernel underneath it.
            if source == EL.EL0 and bit_at(registers.hcr_el2, 27):
                target = EL.EL2
            else:
                target = EL.EL1 if source in (EL.EL0, EL.EL1) else source
        elif getattr(exception, 'from_stage2', False):
            # A stage 2 fault is the hypervisor's business by definition: the guest has
            # no visibility of the tables that produced it.
            target = EL.EL2
        else:
            target = EL.EL1 if source in (EL.EL0, EL.EL1) else source
            if source == EL.EL0 and bit_at(registers.hcr_el2, 27):
                target = EL.EL2

        # Never below the level that raised it, and never above what is implemented.
        if target.value < source.value:
            target = source
        while target.value > self.highest_el.value:
            target = EL(target.value - 1)
        return target

    def _async_target_el(self, kind, source):
        registers = self.registers
        # SCR_EL3 bit 1 is IRQ, bit 2 FIQ, bit 3 EA (SError and aborts).
        scr_bit = {ExceptionType.IRQ: 1, ExceptionType.FIQ: 2, ExceptionType.SERROR: 3}[kind]
        if bit_at(registers.scr_el3, scr_bit) and self.highest_el == EL.EL3:
            return EL.EL3
        # HCR_EL2 bit 4 is IMO, bit 3 FMO, bit 5 AMO.
        hcr_bit = {ExceptionType.IRQ: 4, ExceptionType.FIQ: 3, ExceptionType.SERROR: 5}[kind]
        if source in (EL.EL0, EL.EL1) and bit_at(registers.hcr_el2, hcr_bit):
            return EL.EL2
        if source in (EL.EL0, EL.EL1):
            return EL.EL1
        return source

    def take_exception(self, exception, target_el=None, preferred_return=None):
        """
        AArch64.TakeException - save return state, update PSTATE, vector to the handler.
        """
        if target_el is None:
            target_el = self.target_el_for(exception)
        if preferred_return is None:
            preferred_return = self.registers.get_pc()
        self.exception_count += 1

        saved = self.registers.pstate.to_spsr()
        if self.registers.using_aarch32():
            # Bit 5 of an AArch32 SPSR is the T bit. Without it an ERET back into
            # AArch32 would resume in the wrong instruction set.
            saved |= self.aarch32.registers.cpsr.t << 5
        self.registers.spsr[target_el] = saved
        self.registers.elr[target_el] = preferred_return
        self.registers.esr[target_el] = self.exception_syndrome(exception, target_el)

        if exception.exception_type in (ExceptionType.DATA_ABORT, ExceptionType.INSTRUCTION_ABORT,
                                        ExceptionType.PC_ALIGNMENT):
            self.registers.far[target_el] = lower_chunk(getattr(exception, 'address', 0), 64)

        # An exception loses the reservation: the architecture permits it, and it stops
        # a half-finished critical section from resuming as though nothing happened.
        self.exclusive_monitor.clear(self.cpu_id)
        # Entry is a context synchronisation event, so pending stores are made visible
        # and loads are ordered before the handler runs.
        self.drain_store_buffer()
        self.synchronize_reads()

        offset = self.exception_vector_offset(target_el, exception.exception_type)

        # Entry is always into AArch64 with SP_ELx selected and all exceptions masked.
        self.registers.pstate.el = target_el
        self.registers.pstate.n_rw = 0
        self.registers.pstate.sp = 1
        self.registers.pstate.d = 1
        self.registers.pstate.a = 1
        self.registers.pstate.i = 1
        self.registers.pstate.f = 1
        self.registers.pstate.il = 0
        self.registers.pstate.ss = 0

        self.registers.branch_to(lower_chunk(self.registers.vbar[target_el] + offset, 64))

    def exception_return(self, new_pc, spsr):
        """
        AArch64.ExceptionReturn - restore PSTATE from the SPSR and branch.

        The SPSR carries the exception level to return to, so this is also how execution
        drops from EL3 to EL2 or EL1 during boot: firmware writes the level it wants
        into SPSR_EL3, points ELR_EL3 at the entry point and executes ERET.
        """
        self.registers.pstate.from_spsr(spsr)
        self.registers.branch_to(lower_chunk(new_pc, 64))
        if self.registers.using_aarch32():
            # The AArch32 view keeps its own PC and instruction set, so returning into
            # it means putting both back rather than only setting the AArch64 PC.
            from armulator.armv6.enums import InstrSet as A32InstrSet
            view = self.aarch32
            view.pc = lower_chunk(new_pc, 32)
            view.instruction_set = (A32InstrSet.THUMB if bit_at(spsr, 5)
                                    else A32InstrSet.ARM)
            view.it_state = 0
        # Translation and reservations belong to the level being left behind.
        self.mmu.flush()
        self.drain_store_buffer()
        self.synchronize_reads()

    # ------------------------------------------------------------------
    # Asynchronous exceptions
    # ------------------------------------------------------------------

    def irq_masked(self):
        return bool(self.registers.pstate.i)

    def fiq_masked(self):
        return bool(self.registers.pstate.f)

    def take_physical_irq_exception(self):
        """
        Deliver a physical IRQ. Asynchronous exceptions are taken between instructions, so
        the preferred return is the instruction that would have run next - which is where
        the PC already points by the time the board polls the interrupt controller.
        """
        self.take_exception(IRQException(), None, self.registers.get_pc())

    def take_physical_fiq_exception(self):
        self.take_exception(FIQException(), None, self.registers.get_pc())

    # ------------------------------------------------------------------
    # Stubs the opcodes call into
    # ------------------------------------------------------------------

    def clear_event_register(self):
        self.registers.set_event_register(False)

    def event_registered(self):
        return self.registers.get_event_register()

    def send_event_local(self):
        self.registers.set_event_register(True)

    def send_event(self):
        # SEV is cluster-wide. The core cannot reach its siblings, so it raises a flag
        # the scheduler collects and turns into a wake-up for every core.
        self.registers.set_event_register(True)
        self.event_signalled = True

    def get_and_clear_event_signal(self):
        """Whether this core executed SEV since the last check."""
        signalled = self.event_signalled
        self.event_signalled = False
        return signalled

    def wait_for_event(self):
        # WFE returns immediately if an event is already pending, consuming it. Parking
        # unconditionally would lose the wake-up that arrived just before the WFE.
        if self.registers.get_event_register():
            self.registers.set_event_register(False)
            return
        self.is_wait_for_event = True

    def wait_for_interrupt(self):
        self.is_wait_for_interrupt = True

    def hint_yield(self):
        pass

    def note_deferred_load(self, register, address, size, signed=False,
                           datasize=64, regsize=64):
        """
        Record that a load's value may be settled later than the instruction that
        issued it.

        This is load/store reordering seen from the side a forward simulation can
        actually model. Hoisting a store above a preceding load would mean answering
        that load from the future; delaying the load until after the store is the same
        reordering, and reachable.

        The value already written to the register stands until something reads it. At
        that point it is re-read from memory **only if this core has issued a store since
        the load** - which is the whole structural difference between the two directions
        a load can be reordered:

        * A load with a store after it can be performed late, so the store becomes
          visible first. That is load/store reordering, and the LB litmus test.
        * A load with only other loads after it is reordered the other way, answered from
          before it executed. That is load/load reordering, and the message-passing test.

        A deterministic model cannot apply both to the same load - they move it in
        opposite directions - so which one applies is decided by what actually follows it.
        """
        if not self.load_reorderer.reordering or register == 31:
            return

        issued_after = self.store_count
        current = self.registers._x[register] if register != 31 else 0

        def reload():
            if self.store_count == issued_after:
                # Nothing to be reordered against, so the load stays where it was.
                return current
            data = self.mem_get(address, size)
            if signed:
                data = sign_extend(data, datasize, regsize)
            return data

        self.registers.pending_loads[register] = (address, size, reload)

    def synchronize_reads(self):
        """
        Mark a point after which loads are ordered against everything before it.
        """
        self.load_reorderer.synchronize(self.clock.now)
        # A barrier settles outstanding loads: that is what ordering them means.
        self.registers.resolve_all_pending_loads()

    def data_memory_barrier(self, domain, types):
        # DMB orders accesses rather than forcing completion. Draining is a sound
        # over-approximation: it never permits an ordering DMB would have forbidden.
        self.drain_store_buffer()
        self.synchronize_reads()

    def data_synchronization_barrier(self, domain, types):
        # DSB additionally waits for completion, which is exactly a drain.
        self.drain_store_buffer()
        self.synchronize_reads()

    def instruction_synchronization_barrier(self):
        # Needed before fetching instructions a core has just written.
        self.drain_store_buffer()
        self.synchronize_reads()

    def hint_preload_data(self, address):
        pass

    def hint_preload_instr(self, address):
        pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def this_instr(self):
        return self.opcode

    def this_instr_length(self):
        return self.opcode_len

    def fetch_instruction(self):
        """
        A64 instructions are a fixed 32 bits and must be word aligned.
        """
        pc = self.registers.get_pc()
        if pc != align(pc, 4):
            raise PcAlignmentException(pc, 'PC is not word aligned')
        self.opcode_len = 32
        descriptor = self.translate_address(pc, None, False, 4, True, is_instruction=True)
        self.opcode = self.mem[descriptor, 4]
        return self.opcode

    def decode_instruction(self, instr):
        return op_decode_instruction(instr, self)

    def execute_instruction(self, opcode):
        self.executed_opcode = opcode
        opcode.execute(self)

    def increment_pc_if_needed(self):
        if not self.registers.branch_taken:
            self.registers.increment_pc(4)

    def emulate_cycle(self):
        self.instruction_count += 1
        self.clock.tick()
        # The architected counter is free-running, so it advances whatever the
        # core is doing -- including while it is spinning in a delay loop
        # waiting for exactly this counter to move.
        self.registers.generic_timer.tick()
        self.retire_pending_stores()
        if self.registers.using_aarch32():
            self.emulate_aarch32_cycle()
            return
        self.registers.branch_taken = False
        # The address of the faulting or trapping instruction is the preferred return.
        preferred_return = self.registers.get_pc()
        try:
            if self.registers.pstate.il:
                raise IllegalStateException('PSTATE.IL is set')
            instr = self.fetch_instruction()
            opcode_class = self.decode_instruction(instr)
            if not opcode_class:
                raise UndefinedInstructionException(f'undefined instruction 0x{instr:08X}')
            opcode = opcode_class.from_bitarray(instr, self)
            if opcode is None:
                raise UndefinedInstructionException(f'undefined instruction 0x{instr:08X}')
            self.execute_instruction(opcode)
            self.increment_pc_if_needed()
        except EndOfInstruction:
            self.increment_pc_if_needed()
        except SMCException as exception:
            # With a cluster attached, SMC is the PSCI call firmware uses to bring up
            # secondary cores; it is serviced here rather than vectored, standing in for
            # the secure firmware that would answer it at EL3.
            if self.psci_handler is not None and self.psci_handler(self):
                self.increment_pc_if_needed()
            else:
                self.take_exception(exception, None, lower_chunk(preferred_return + 4, 64))
        except (SVCException, HVCException, SoftwareBreakpointException) as exception:
            # These complete, so the return address is the following instruction.
            self.take_exception(exception, None, lower_chunk(preferred_return + 4, 64))
        except (DataAbortException, InstructionAbortException, PcAlignmentException,
                SpAlignmentException, UndefinedInstructionException, IllegalStateException,
                SystemRegisterTrapException, AdvSimdFpAccessTrapException) as exception:
            # These are faults, so the instruction is re-taken on return.
            self.take_exception(exception, None, preferred_return)
        except ArmV8Exception as exception:
            self.take_exception(exception, None, preferred_return)
