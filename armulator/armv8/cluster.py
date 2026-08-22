"""
A cluster of AArch64 cores.

The Jetson Nano's Tegra X1 carries four Cortex-A57s sharing one memory system, one
interrupt controller and one exclusive monitor. That sharing is the whole point: it is
what makes a spinlock necessary and an IPI possible.

Scheduling is cooperative round-robin - each core runs a slice of instructions, then the
next takes a turn. That is not how real hardware interleaves, and deliberately so: a
deterministic order makes a failing test reproducible, where true concurrency would
not be. The slice size is adjustable, and a smaller slice interleaves more finely, which
is the knob to reach for when checking that a lock actually holds.

Secondary cores start parked. Real firmware releases them through PSCI, and that is
modelled here rather than started-and-spinning, because the release path is exactly where
multi-core bring-up bugs live.
"""

from armulator.armv6.memory_controller_hub import MemoryControllerHub
from armulator.armv8.arm_v8 import ArmV8
from armulator.armv8.exclusive_monitor import ExclusiveMonitor
from armulator.armv8.memory_history import Clock, MemoryHistory
from armulator.armv8.store_buffer import MemoryModel

#: PSCI function identifiers, from the Arm Power State Coordination Interface.
PSCI_VERSION = 0x84000000
PSCI_CPU_OFF = 0x84000002
PSCI_CPU_ON_32 = 0x84000003
PSCI_CPU_ON_64 = 0xC4000003
PSCI_AFFINITY_INFO_32 = 0x84000004
PSCI_AFFINITY_INFO_64 = 0xC4000004

#: PSCI return codes.
PSCI_SUCCESS = 0
PSCI_NOT_SUPPORTED = -1 & 0xFFFFFFFFFFFFFFFF
PSCI_INVALID_PARAMETERS = -2 & 0xFFFFFFFFFFFFFFFF
PSCI_ALREADY_ON = -4 & 0xFFFFFFFFFFFFFFFF

#: Affinity states reported by PSCI_AFFINITY_INFO.
AFFINITY_ON = 0
AFFINITY_OFF = 1


class Cluster:
    """
    Several cores sharing memory, an exclusive monitor and an interrupt controller.

    :param num_cores: how many cores the cluster contains
    :param memory_list: memory map, shared by every core
    :param slice_size: instructions each core runs before the next takes a turn
    """

    def __init__(self, num_cores=4, memory_list=None, slice_size=16):
        self.exclusive_monitor = ExclusiveMonitor()
        self.memory = MemoryControllerHub.from_memory_list(memory_list or [])
        #: One clock for the cluster, so writes and reads on different cores carry
        #: comparable timestamps.
        self.clock = Clock()
        #: What memory used to hold, shared because one core's writes are what another
        #: core's reordered load looks back at.
        self.memory_history = MemoryHistory()
        self.slice_size = slice_size
        self.gic = None
        self.steps = 0

        self.cores = []
        for cpu_id in range(num_cores):
            core = ArmV8(cpu_id=cpu_id, exclusive_monitor=self.exclusive_monitor,
                         memory_history=self.memory_history, clock=self.clock)
            # Every core addresses the same memory: this is what makes them a cluster
            # rather than four unrelated processors.
            core.mem = self.memory
            core.take_reset()
            core.psci_handler = self.handle_psci
            self.cores.append(core)

        #: Core 0 comes out of reset running; the rest wait to be released.
        self.powered_on = [cpu_id == 0 for cpu_id in range(num_cores)]
        #: True once a core has parked on a tight self-branch, the way bare-metal code
        #: signals it is finished.
        self.halted = [False] * num_cores
        self._previous_pc = [None] * num_cores

    # ------------------------------------------------------------------
    # Core state
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.cores)

    def __getitem__(self, cpu_id):
        return self.cores[cpu_id]

    @property
    def primary(self):
        return self.cores[0]

    def is_running(self, cpu_id) -> bool:
        """
        A core runs when it is powered on, not parked in WFI or WFE, and not spinning
        on a halt loop.
        """
        core = self.cores[cpu_id]
        return (self.powered_on[cpu_id]
                and not self.halted[cpu_id]
                and not core.is_wait_for_interrupt
                and not core.is_wait_for_event)

    def power_on(self, cpu_id, entry_point, context_id=0) -> None:
        """
        Release a secondary core at ``entry_point``.

        The context argument arrives in X0, which is how firmware hands a per-core stack
        or data pointer to a core it has just woken.
        """
        core = self.cores[cpu_id]
        core.take_reset()
        core.registers.set_x(0, context_id)
        core.registers.branch_to(entry_point)
        core.registers.branch_taken = False
        core.is_wait_for_interrupt = False
        core.is_wait_for_event = False
        self.powered_on[cpu_id] = True
        self.halted[cpu_id] = False
        self._previous_pc[cpu_id] = None

    def power_off(self, cpu_id) -> None:
        self.powered_on[cpu_id] = False

    def core_for_mpidr(self, mpidr):
        """
        Find the core an MPIDR value names. Only Aff0 varies in a single cluster.
        """
        target = mpidr & 0xFF
        cluster = (mpidr >> 8) & 0xFF
        if cluster != 0 or target >= len(self.cores):
            return None
        return target

    # ------------------------------------------------------------------
    # PSCI
    # ------------------------------------------------------------------

    def handle_psci(self, core) -> bool:
        """
        Service a PSCI call made with SMC. Returns True when the call was recognised.

        This stands in for the secure firmware that would normally sit at EL3. Modelling
        it means secondary cores are released the way real firmware releases them, rather
        than by the test harness reaching in and setting a PC.
        """
        function = core.registers.get_x(0) & 0xFFFFFFFF

        if function == (PSCI_VERSION & 0xFFFFFFFF):
            # Version 1.0, as major:minor in the two halves of the word.
            core.registers.set_x(0, (1 << 16) | 0)
            return True

        if function in (PSCI_CPU_ON_32 & 0xFFFFFFFF, PSCI_CPU_ON_64 & 0xFFFFFFFF):
            target = self.core_for_mpidr(core.registers.get_x(1))
            entry_point = core.registers.get_x(2)
            context_id = core.registers.get_x(3)
            if target is None:
                core.registers.set_x(0, PSCI_INVALID_PARAMETERS)
            elif self.powered_on[target]:
                core.registers.set_x(0, PSCI_ALREADY_ON)
            else:
                self.power_on(target, entry_point, context_id)
                core.registers.set_x(0, PSCI_SUCCESS)
            return True

        if function == (PSCI_CPU_OFF & 0xFFFFFFFF):
            self.power_off(core.cpu_id)
            core.registers.set_x(0, PSCI_SUCCESS)
            return True

        if function in (PSCI_AFFINITY_INFO_32 & 0xFFFFFFFF, PSCI_AFFINITY_INFO_64 & 0xFFFFFFFF):
            target = self.core_for_mpidr(core.registers.get_x(1))
            if target is None:
                core.registers.set_x(0, PSCI_INVALID_PARAMETERS)
            else:
                core.registers.set_x(
                    0, AFFINITY_ON if self.powered_on[target] else AFFINITY_OFF
                )
            return True

        core.registers.set_x(0, PSCI_NOT_SUPPORTED)
        return True

    # ------------------------------------------------------------------
    # Interrupts
    # ------------------------------------------------------------------

    def deliver_interrupts(self, cpu_id) -> bool:
        """
        Offer a pending interrupt to one core. Returns True if it took an exception.
        """
        if self.gic is None:
            return False
        core = self.cores[cpu_id]
        self.gic.current_cpu = cpu_id
        self.gic.refresh()
        asserting = self.gic.irq_pending_for(cpu_id)

        if asserting:
            # An interrupt wakes a core out of WFI even when it is masked, because the
            # wake-up condition is the interrupt arriving, not it being taken.
            core.is_wait_for_interrupt = False

        if not asserting or core.irq_masked():
            return False
        core.take_physical_irq_exception()
        return True

    def send_event_to_all(self, sender_id) -> None:
        """SEV wakes every core waiting on an event, which is how WFE loops make progress."""
        for cpu_id, core in enumerate(self.cores):
            core.registers.set_event_register(True)
            if cpu_id != sender_id:
                core.is_wait_for_event = False

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def step_core(self, cpu_id) -> int:
        """
        Run one core for a slice. Returns the number of instructions executed.
        """
        core = self.cores[cpu_id]
        if not self.powered_on[cpu_id]:
            return 0

        if self.gic is not None:
            self.gic.current_cpu = cpu_id

        # Offered before the running check: an interrupt is precisely what brings a core
        # out of WFI, so a parked core still has to be given the chance to take one.
        self.deliver_interrupts(cpu_id)

        if not self.is_running(cpu_id):
            # A core that is halted or waiting still drains its store buffer: pending
            # writes do not depend on the core doing anything further, and leaving them
            # stranded would stall a reader forever and look like an ordering bug.
            # They retire one at a time rather than all at once, so another core can
            # still observe the state in between - dumping the whole buffer atomically
            # would hide exactly the reordering this models.
            core.instruction_count += 1
            self.clock.tick()
            core.retire_pending_stores()
            return 0

        executed = 0
        for _ in range(self.slice_size):
            self.deliver_interrupts(cpu_id)
            if not self.is_running(cpu_id):
                break
            pc = core.registers.get_pc()
            exceptions_before = core.exception_count
            core.emulate_cycle()
            executed += 1
            # A core sitting on `b .` has finished. An exception taken this step means
            # the PC repeated for a different reason - faulting round the vector table -
            # which is not the same thing and must not look like completion.
            if (core.registers.get_pc() == pc and self._previous_pc[cpu_id] == pc
                    and core.exception_count == exceptions_before):
                self.halted[cpu_id] = True
                break
            self._previous_pc[cpu_id] = pc
            # An SEV anywhere releases every waiting core, so it is handled centrally
            # rather than left to the core that executed it.
            if core.get_and_clear_event_signal():
                self.send_event_to_all(cpu_id)
            if not self.powered_on[cpu_id]:
                break
        return executed

    def step_round(self) -> int:
        """One pass over every core."""
        executed = 0
        for cpu_id in range(len(self.cores)):
            executed += self.step_core(cpu_id)
        self.steps += 1
        return executed

    @property
    def all_parked(self) -> bool:
        """True when no core can make progress without outside help."""
        return not any(self.is_running(cpu_id) for cpu_id in range(len(self.cores)))

    @property
    def all_halted(self) -> bool:
        """True when every powered-on core has parked on a halt loop."""
        live = [c for c in range(len(self.cores)) if self.powered_on[c]]
        return bool(live) and all(self.halted[c] for c in live)

    def set_memory_model(self, model, latency=None) -> None:
        """
        Set the memory model on every core.

        :param model: a :class:`~armulator.armv8.store_buffer.MemoryModel`
        :param latency: how many instructions a store may sit unpublished, widening or
            narrowing the window in which a reordering can be observed

        The default is sequential consistency, which is stronger than any real machine.
        Switching to ``ADVERSARIAL`` is how you find out whether lock-free code depends
        on ordering it never actually established.
        """
        for core in self.cores:
            core.set_memory_model(model)
            if latency is not None:
                core.store_buffer.latency = latency

    @property
    def memory_model(self):
        return self.cores[0].store_buffer.model

    def drain_all(self) -> None:
        """Settle every pending store, so observed memory reflects what was written."""
        for core in self.cores:
            core.drain_store_buffer()

    def run(self, max_instructions=10000) -> int:
        """
        Run until every core parks or the budget is spent.
        """
        total = 0
        budget = max_instructions
        while budget > 0:
            executed = self.step_round()
            total += executed
            budget -= max(executed, 1)
            if self.all_parked:
                break
        # Leaving stores queued would make the final memory state depend on where the
        # run happened to stop, which is not something a test should have to reason about.
        self.drain_all()
        return total

    def run_until(self, predicate, max_instructions=10000) -> bool:
        """
        Run until ``predicate()`` holds. Returns whether it did.
        """
        budget = max_instructions
        if predicate():
            return True
        while budget > 0:
            executed = self.step_round()
            if predicate():
                return True
            budget -= max(executed, 1)
            if self.all_parked:
                break
        return False

    def format_state(self) -> str:
        lines = []
        for cpu_id, core in enumerate(self.cores):
            if not self.powered_on[cpu_id]:
                state = 'off'
            elif self.halted[cpu_id]:
                state = 'halted'
            elif core.is_wait_for_interrupt:
                state = 'WFI'
            elif core.is_wait_for_event:
                state = 'WFE'
            else:
                state = 'running'
            lines.append(f'cpu{cpu_id}: {state:8} pc=0x{core.registers.get_pc():X}')
        return '\n'.join(lines)
