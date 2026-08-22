"""
CPU adapters for the board layer.

A :class:`~armulator.boards.Board` needs only a handful of things from its processor:
somewhere to hang the memory map, a way to reset it, a way to point it at an address, a
way to step it, and a way to deliver an interrupt. Those five things are spelled
differently in AArch32 and AArch64 - ``CPSR.I`` versus ``PSTATE.I``, ``SCTLR.M`` versus
``SCTLR_EL1.M``, an instruction set to select versus none at all.

Rather than scatter conditionals through ``Board``, each architecture gets an adapter
exposing one vocabulary. Adding a third architecture later means adding one class here
and nothing else.
"""

from abc import ABC, abstractmethod

from armulator.armv6.memory_controller_hub import MemoryController


class CpuAdapter(ABC):
    """
    Uniform interface over a processor model.
    """

    #: Human readable architecture name, used in diagnostics.
    name = 'unknown'
    #: Width of one instruction in bytes, for halt detection and disassembly.
    instruction_size = 4
    #: True when the architecture has a second, narrower instruction set.
    supports_thumb = False

    def __init__(self, cpu):
        self.cpu = cpu

    # ------------------------------------------------------------------
    # Memory map
    # ------------------------------------------------------------------

    @property
    def memories(self):
        return self.cpu.mem.memories

    def set_memories(self, controllers) -> None:
        self.cpu.mem.memories = list(controllers)

    def map(self, device, base, size) -> None:
        self.cpu.mem.memories.append(MemoryController(device, base, base + size))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def set_flat_addressing(self) -> None:
        """
        Disable address translation so firmware sees physical addresses, as at reset.
        """

    @abstractmethod
    def reset(self) -> None:
        """Take reset."""

    @abstractmethod
    def set_pc(self, address, thumb=False) -> None:
        """Point the processor at ``address``."""

    @property
    @abstractmethod
    def pc(self) -> int:
        """Address of the instruction about to execute."""

    def step(self) -> None:
        self.cpu.emulate_cycle()

    # ------------------------------------------------------------------
    # Interrupts
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def interrupts_masked(self) -> bool:
        """True when the processor is not currently accepting physical IRQs."""

    @abstractmethod
    def take_irq(self) -> None:
        """Deliver a physical IRQ to the processor."""

    @property
    def exception_count(self) -> int:
        """
        Exceptions taken since reset, where the model tracks it.

        A board uses this to tell a firmware parked on its halt loop from one faulting
        repeatedly through the vector table: both leave the PC unchanged step to step.
        Models that do not count return 0, and the board falls back to plain PC matching.
        """
        return getattr(self.cpu, 'exception_count', 0)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    @abstractmethod
    def assemble(self, source, address=0, thumb=False) -> bytes:
        """Assemble ``source`` for this architecture."""

    def __repr__(self):
        return f'<{type(self).__name__} {self.name} pc=0x{self.pc:X}>'


class ArmV6Adapter(CpuAdapter):
    """
    AArch32 (A32/T32) via the ARMv6 core.
    """

    name = 'armv6'
    supports_thumb = True

    def __init__(self, cpu=None):
        from armulator.armv6.arm_v6 import ArmV6
        super().__init__(cpu if cpu is not None else ArmV6())

    def set_flat_addressing(self) -> None:
        self.cpu.registers.sctlr.m = 0

    def reset(self) -> None:
        self.cpu.take_reset()

    def set_pc(self, address, thumb=False) -> None:
        # The instruction set is selected explicitly rather than inherited from SCTLR.TE,
        # which resets to Thumb in armulator's default configuration and would silently
        # misdecode A32 firmware.
        from armulator.armv6.enums import InstrSet
        self.cpu.registers.select_instr_set(InstrSet.THUMB if thumb else InstrSet.ARM)
        self.cpu.registers.branch_to(address)

    @property
    def pc(self) -> int:
        return self.cpu.registers.pc_store_value()

    @property
    def interrupts_masked(self) -> bool:
        return bool(self.cpu.registers.cpsr.i)

    def take_irq(self) -> None:
        # In the AArch32 model exception entry is a method on the register file.
        self.cpu.registers.take_physical_irq_exception()

    def assemble(self, source, address=0, thumb=False) -> bytes:
        from armulator.boards.firmware import assemble
        return assemble(source, thumb=thumb, address=address)


class ArmV8Adapter(CpuAdapter):
    """
    AArch64 via the ARMv8 core, as found on the Cortex-A57 and friends.
    """

    name = 'armv8'
    supports_thumb = False

    def __init__(self, cpu=None):
        from armulator.armv8.arm_v8 import ArmV8
        super().__init__(cpu if cpu is not None else ArmV8())

    def set_flat_addressing(self) -> None:
        # Clear SCTLR_EL1.M so firmware sees physical addresses at reset. Firmware is
        # free to build page tables and set it again; translation is implemented.
        self.cpu.registers.sctlr_el1 = self.cpu.registers.sctlr_el1 & ~1

    def reset(self) -> None:
        self.cpu.take_reset()
        # take_reset restores architectural reset values, which turn the MMU back on for
        # some configurations, so flat addressing is reapplied after it.
        self.set_flat_addressing()

    def set_pc(self, address, thumb=False) -> None:
        if thumb:
            raise ValueError('AArch64 has no Thumb instruction set; assemble as A64')
        self.cpu.registers.branch_to(address)

    @property
    def pc(self) -> int:
        return self.cpu.registers.get_pc()

    @property
    def interrupts_masked(self) -> bool:
        return bool(self.cpu.registers.pstate.i)

    def take_irq(self) -> None:
        # In the AArch64 model exception entry needs the CPU, since it consults the
        # vector base and the current exception level together.
        self.cpu.take_physical_irq_exception()

    def assemble(self, source, address=0, thumb=False) -> bytes:
        if thumb:
            raise ValueError('AArch64 has no Thumb instruction set; assemble as A64')
        from armulator.boards.firmware import assemble_a64
        return assemble_a64(source, address=address)


#: Adapters by architecture name, for boards that select their CPU by string.
ADAPTERS = {
    'armv6': ArmV6Adapter,
    'armv8': ArmV8Adapter,
}


def make_adapter(arch):
    """
    Build an adapter from an architecture name, an adapter instance, or an adapter class.
    """
    if isinstance(arch, CpuAdapter):
        return arch
    if isinstance(arch, type) and issubclass(arch, CpuAdapter):
        return arch()
    try:
        return ADAPTERS[arch]()
    except KeyError:
        raise ValueError(
            f'unknown architecture {arch!r}; expected one of {sorted(ADAPTERS)}'
        ) from None
