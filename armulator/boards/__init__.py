"""
Board models: a CPU plus a peripheral memory map at the real addresses.

Each board wires peripherals at the physical addresses the actual SoC uses,
so firmware written against the published datasheets -- or lifted from a
real driver -- lands on the right registers without modification.

    board = RaspberryPi4(trace=True)
    board.load(board.CODE_BASE, firmware_bytes)
    board.run(2000)
    assert board.gpio.level(17) is True

SCOPE NOTE
----------
The Pi 3, Pi 4 and Jetson Nano all use ARMv8-A cores.  Boards select their
processor with ``arch=``, defaulting to the ARMv6 core (A32/T32, integer
only) which is what the peripheral test firmware in this repository is
written against.  Passing ``arch='armv8'`` -- or using one of the ``*A64``
board classes -- builds the board around the AArch64 core instead, and
``cores=`` (or ``JetsonNanoA64Smp``) gives a multi-core cluster.

Both cores drive the same peripheral models at the same addresses, so A32
test firmware exercises exactly the register sequences an AArch64 driver
performs.  Reach for the AArch64 boards when the code under test is itself
AArch64, or when it depends on translation, several cores, or memory
ordering.  See AARCH64.md.

Neither core boots a vendor kernel: only a handful of each SoC's peripherals
are modelled, which is nowhere near enough for one to get started.
"""

from armulator.armv6.memory_controller_hub import MemoryController
from armulator.armv6.memory_types import RAM
from armulator.boards.cpu import make_adapter
from armulator.peripherals.gpio_bcm import BcmGpio
from armulator.armv8.generic_timer import TIMER_PPI
from armulator.peripherals.gic400 import SPI_BASE, Gic400
from armulator.peripherals.gpio_tegra import TegraGpio
from armulator.peripherals.serial_bus import Bcm2835I2c, Bcm2835Spi
from armulator.peripherals.spi_slave import Bcm2835SpiSlave
from armulator.peripherals.spi_tegra import Tegra210Spi
from armulator.peripherals.uart_8250 import TegraUart
from armulator.peripherals.uart_pl011 import BcmSystemTimer, Pl011Uart


class Board:
    """
    Base class: a CPU with RAM and a set of memory-mapped peripherals.

    :param ram_size: bytes of RAM mapped at :attr:`RAM_BASE`
    :param trace: record every peripheral register access
    """

    #: Physical base of the peripheral window.
    PERIPHERAL_BASE = 0x00000000
    #: Where :meth:`load` puts code by default.
    CODE_BASE = 0x00008000
    RAM_BASE = 0x00000000
    #: Default processor architecture; override in a subclass or pass ``arch=``.
    ARCH = 'armv6'
    #: Default number of cores; override in a subclass or pass ``cores=``.
    CORES = 1

    def __init__(self, ram_size=0x100000, trace=False, arch=None, cores=None):
        arch = self.ARCH if arch is None else arch
        cores = self.CORES if cores is None else cores

        #: The cluster, when this board has more than one core.
        self.cluster = None
        if cores > 1:
            if arch != 'armv8':
                raise ValueError('multiple cores are only modelled for the armv8 core')
            from armulator.armv8.cluster import Cluster
            # Every core shares one memory hub, so peripherals attached below are
            # visible to all of them without any further wiring.
            self.cluster = Cluster(cores, [])
            self.cpu_adapter = make_adapter(arch)
            self.cpu_adapter.cpu = self.cluster.primary
        else:
            #: Adapter presenting a uniform interface over the processor model.
            self.cpu_adapter = make_adapter(arch)
        #: The underlying processor model. On a cluster this is the primary core.
        self.cpu = self.cpu_adapter.cpu
        self.trace = trace
        self.devices = {}
        #: Interrupt controller, if this board has one.
        self.gic = None
        #: True once firmware has parked on its halt loop.
        self.halted = False
        #: True when the CPU was last seen faulting repeatedly through the vector
        #: table rather than parking cleanly.
        self.fault_loop = False

        # Flat physical addressing: the MPU/MMU is off, so firmware sees
        # the peripheral map directly, as it does at reset on real hardware.
        self.cpu_adapter.set_flat_addressing()

        self.ram = RAM(ram_size)
        self.cpu_adapter.set_memories([
            MemoryController(self.ram, self.RAM_BASE, self.RAM_BASE + ram_size)
        ])
        self._build()
        if self.cluster is not None:
            self.cluster.gic = self.gic
            # Reset every core, then reapply flat addressing: reset restores the
            # architectural SCTLR value, which turns translation back on.
            for core in self.cluster.cores:
                core.take_reset()
                core.registers.sctlr_el1 = core.registers.sctlr_el1 & ~1
        else:
            self.cpu_adapter.reset()

    @property
    def arch(self) -> str:
        """Architecture name of the processor this board was built around."""
        return self.cpu_adapter.name

    @property
    def cores(self):
        """Every core on this board, primary first."""
        return self.cluster.cores if self.cluster is not None else [self.cpu]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self):
        """Subclasses attach their peripherals here."""

    def attach(self, name, device, offset=None, address=None):
        """
        Map ``device`` into the physical address space.

        Give either ``offset`` (relative to :attr:`PERIPHERAL_BASE`) or an
        absolute ``address``.
        """
        if (offset is None) == (address is None):
            raise ValueError('pass exactly one of offset= or address=')
        base = self.PERIPHERAL_BASE + offset if address is None else address
        device.trace = self.trace
        self.cpu_adapter.map(device, base, device.size)
        self.devices[name] = device
        setattr(self, name, device)
        return device

    def connect_irq(self, device, spi):
        """
        Wire ``device``'s interrupt output to SPI number ``spi`` on the
        board's GIC.  Returns the resulting interrupt ID.
        """
        if self.gic is None:
            raise RuntimeError(f'{type(self).__name__} has no interrupt controller')
        return self.gic.connect(device, spi)

    # ------------------------------------------------------------------
    # Running firmware
    # ------------------------------------------------------------------
    def load(self, address, code) -> None:
        """Copy ``code`` into RAM at physical ``address``."""
        offset = address - self.RAM_BASE
        if offset < 0 or offset + len(code) > self.ram.size:
            raise ValueError(f'0x{address:X} + {len(code)} bytes is outside RAM')
        self.ram.write(offset, len(code), code)

    def start(self, address=None, thumb=False) -> None:
        """
        Point the PC at ``address`` (default :attr:`CODE_BASE`).

        On AArch32 the instruction set is selected explicitly rather than
        inherited from SCTLR.TE, which resets to Thumb in armulator's default
        configuration and would silently misdecode A32 firmware.  AArch64 has
        no second instruction set, so ``thumb=True`` is rejected there.
        """
        self.halted = False
        self.fault_loop = False
        if self.cluster is not None:
            # Only the primary comes out of reset running; secondaries wait for PSCI.
            self.cluster.power_on(0, self.CODE_BASE if address is None else address)
            return
        self.cpu_adapter.set_pc(
            self.CODE_BASE if address is None else address, thumb=thumb
        )

    def step(self) -> None:
        """Execute one instruction and deliver any pending peripheral IRQ."""
        if self.cluster is not None:
            self.cluster.step_round()
            return
        self.cpu.emulate_cycle()
        self.service_interrupts()

    def run(self, max_instructions=10000, stop_at=None) -> int:
        """
        Run until ``stop_at`` is reached or ``max_instructions`` execute.

        Returns the number of instructions actually executed.  Also stops
        early if the CPU parks on a tight self-branch (``B .``), the usual
        way bare-metal test firmware signals that it is done.

        A CPU faulting repeatedly through the vector table also leaves the PC
        unchanged from step to step, so that case is told apart by watching
        whether an exception was taken; it sets :attr:`fault_loop` rather than
        :attr:`halted`, so a crashing firmware is not mistaken for a finished one.
        """
        if self.cluster is not None:
            executed = self.cluster.run(max_instructions)
            self.halted = self.cluster.all_halted
            self.fault_loop = False
            return executed

        executed = 0
        previous_pc = None
        for _ in range(max_instructions):
            pc = self.cpu_adapter.pc
            if stop_at is not None and pc == stop_at:
                break
            exceptions_before = self.cpu_adapter.exception_count
            self.step()
            executed += 1
            new_pc = self.cpu_adapter.pc
            if new_pc == pc and pc == previous_pc:
                if self.cpu_adapter.exception_count != exceptions_before:
                    # Faulting round the vector table, not parked on a halt loop.
                    self.fault_loop = True
                    self.halted = False
                    return executed
                # Spinning on B . -- firmware has finished.  Recorded so a
                # scheduler can tell a parked board from a busy one; two
                # instructions still elapse here, so callers must not rely
                # on a zero return to detect it.
                self.halted = True
                return executed
            previous_pc = pc
        self.halted = False
        return executed

    # ------------------------------------------------------------------
    # Interrupts
    # ------------------------------------------------------------------
    def sample_timer(self) -> None:
        """
        Drive the EL1 physical timer's PPI from the core's own timer.

        The generic timer is part of the core rather than a peripheral, so it
        is not in :attr:`devices` and ``Gic400.refresh`` does not see it. It
        arrives as PPI 30, which is private to a core -- this model keeps one
        line per interrupt ID rather than one per core, so on a cluster the
        primary's timer drives it. The ARMv6 core has no architected timer and
        is skipped.
        """
        timer = getattr(self.cpu.registers, 'generic_timer', None)
        if timer is not None:
            self.gic.set_line(TIMER_PPI, timer.irq_pending)

    def pending_irq(self):
        """Names of devices currently asserting their IRQ line."""
        return [n for n, d in self.devices.items()
                if d is not self.gic and d.irq_pending]

    def service_interrupts(self) -> bool:
        """
        Deliver a physical IRQ to the CPU if an interrupt is asserting and
        interrupts are unmasked.  Returns True if an exception was taken.

        When the board has a GIC, device lines are sampled into the
        distributor and the GIC's own output decides delivery -- so
        enable, priority and mask settings are all honoured.  Boards
        without a GIC fall back to polling device lines directly.
        """
        if self.gic is not None:
            self.sample_timer()
            self.gic.refresh()
            asserting = self.gic.irq_pending
        else:
            asserting = bool(self.pending_irq())
        if self.cpu_adapter.interrupts_masked:
            return False
        if not asserting:
            return False
        self.cpu_adapter.take_irq()
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def format_trace(self) -> str:
        return '\n'.join(
            d.format_trace() for d in self.devices.values() if d.accesses
        )


class RaspberryPi3(Board):
    """
    Raspberry Pi 3 (BCM2837).  Peripherals at 0x3F000000.

    Real core is Cortex-A53 (ARMv8-A); see the scope note in this module.
    """

    PERIPHERAL_BASE = 0x3F000000
    CODE_BASE = 0x00008000

    GPIO_OFFSET = 0x200000
    UART0_OFFSET = 0x201000
    SYSTIMER_OFFSET = 0x003000
    SPI0_OFFSET = 0x204000
    I2C1_OFFSET = 0x804000
    SPI_SLAVE_OFFSET = 0x214000

    def _build(self):
        self.attach('gpio', BcmGpio(pull_style='legacy', name='gpio'),
                    offset=self.GPIO_OFFSET)
        self.attach('uart', Pl011Uart(name='uart0'), offset=self.UART0_OFFSET)
        self.attach('timer', BcmSystemTimer(name='systimer'),
                    offset=self.SYSTIMER_OFFSET)
        self.attach('spi', Bcm2835Spi(name='spi0'), offset=self.SPI0_OFFSET)
        self.attach('i2c', Bcm2835I2c(name='i2c1'), offset=self.I2C1_OFFSET)
        self.attach('spi_slave', Bcm2835SpiSlave(name='spi_slave'),
                    offset=self.SPI_SLAVE_OFFSET)


class RaspberryPi4(Board):
    """
    Raspberry Pi 4 (BCM2711) in low-peripheral mode.  Peripherals at
    0xFE000000, and the BCM2711 pull up/down register scheme.

    Real core is Cortex-A72 (ARMv8-A); see the scope note in this module.
    """

    PERIPHERAL_BASE = 0xFE000000
    CODE_BASE = 0x00008000

    GPIO_OFFSET = 0x200000
    UART0_OFFSET = 0x201000
    SYSTIMER_OFFSET = 0x003000
    SPI0_OFFSET = 0x204000
    I2C1_OFFSET = 0x804000
    SPI_SLAVE_OFFSET = 0x214000

    #: The BCM2711's GIC-400 sits outside the legacy peripheral window.
    GIC_ADDRESS = 0xFF840000

    #: SPI numbers used by :meth:`_build` when wiring the GIC.
    GPIO_SPI = 113
    UART_SPI = 121
    SPI0_SPI = 118
    I2C1_SPI = 117

    def _build(self):
        self.attach('gpio', BcmGpio(pull_style='bcm2711', name='gpio'),
                    offset=self.GPIO_OFFSET)
        self.attach('uart', Pl011Uart(name='uart0'), offset=self.UART0_OFFSET)
        self.attach('timer', BcmSystemTimer(name='systimer'),
                    offset=self.SYSTIMER_OFFSET)
        self.attach('spi', Bcm2835Spi(name='spi0'), offset=self.SPI0_OFFSET)
        self.attach('i2c', Bcm2835I2c(name='i2c1'), offset=self.I2C1_OFFSET)
        self.attach('spi_slave', Bcm2835SpiSlave(name='spi_slave'),
                    offset=self.SPI_SLAVE_OFFSET)

        self.gic = self.attach('gic', Gic400(name='gic400'),
                               address=self.GIC_ADDRESS)
        self.connect_irq(self.gpio, self.GPIO_SPI)
        self.connect_irq(self.uart, self.UART_SPI)
        self.connect_irq(self.spi, self.SPI0_SPI)
        self.connect_irq(self.i2c, self.I2C1_SPI)


class JetsonNano(Board):
    """
    NVIDIA Jetson Nano (Tegra X1 / T210).  GPIO controller at 0x6000D000.

    Real core is Cortex-A57 (ARMv8-A); see the scope note in this module.
    Only the GPIO block, SPI1, the GIC and the debug UART are modelled -- the
    Tegra boot ROM and CBoot chain are proprietary and out of scope.

    The console is UART-A, a 16550 with 4-byte register spacing
    (:class:`~armulator.peripherals.uart_8250.TegraUart`), not a PL011.
    """

    PERIPHERAL_BASE = 0x60000000
    CODE_BASE = 0x80080000
    RAM_BASE = 0x80000000

    GPIO_ADDRESS = 0x6000D000
    UARTA_ADDRESS = 0x70006000
    SPI_ADDRESS = 0x7000D400

    #: Tegra X1's GIC base.  The GIC-400 block puts its distributor at
    #: +0x1000 and its CPU interface at +0x2000, so this is 0x50040000 and
    #: *not* the 0x50041000 the datasheets and device tree quote -- those
    #: name the distributor.  Getting this wrong shifts every GIC register
    #: by 0x1000, which reads back as zero rather than faulting: the
    #: distributor simply never enables and no interrupt is ever delivered.
    GIC_ADDRESS = 0x50040000
    #: Distributor and CPU interface, for firmware that wants the addresses.
    GICD_ADDRESS = 0x50041000
    GICC_ADDRESS = 0x50042000

    GPIO_SPI = 32
    UART_SPI = 36
    SPI_SPI = 39

    def _build(self):
        self.attach('gpio', TegraGpio(name='tegra_gpio'), address=self.GPIO_ADDRESS)
        # UART-A is a 16550 with 4-byte register spacing, not a PL011. The
        # two share offset 0 for the data register and disagree about
        # everything else, so a PL011 here accepts the first character
        # written and then hangs any driver that polls LSR for THRE.
        self.attach('uart', TegraUart(name='uarta'), address=self.UARTA_ADDRESS)
        # SPI1 at 0x7000d400, the controller the Jetson's 40-pin header
        # exposes.  Register map follows the in-tree tegra114 driver, which
        # covers T210.  The Jetson still has no modelled SPI *slave*, so
        # cross-device SPI should use a Pi as the slave end.  See JETSON.md.
        self.attach('spi', Tegra210Spi(name='spi1'), address=self.SPI_ADDRESS)

        self.gic = self.attach('gic', Gic400(name='gic500'),
                               address=self.GIC_ADDRESS)
        self.connect_irq(self.gpio, self.GPIO_SPI)
        self.connect_irq(self.uart, self.UART_SPI)
        self.connect_irq(self.spi, self.SPI_SPI)


class JetsonNanoA64(JetsonNano):
    """
    Jetson Nano built around the AArch64 core, which is what the Tegra X1's
    Cortex-A57 cluster actually runs.

    Same peripheral map as :class:`JetsonNano` -- only the processor differs.
    Firmware must be assembled as A64 (see
    :func:`armulator.boards.firmware.firmware_a64`).

    Loads, stores and branches against the peripheral map all execute, so A64
    firmware runs here end to end -- an earlier version of this docstring said
    otherwise and was left behind by the core's progress.
    """

    ARCH = 'armv8'


class JetsonNanoA64Smp(JetsonNanoA64):
    """
    Jetson Nano with the full quad-core Cortex-A57 cluster the Tegra X1 actually carries.

    Core 0 starts at :attr:`CODE_BASE`; the other three wait for a PSCI ``CPU_ON`` call,
    which is how real firmware releases them.
    """

    CORES = 4


class RaspberryPi3A64(RaspberryPi3):
    """
    Raspberry Pi 3 built around the AArch64 core, matching its Cortex-A53.
    See the caveat on :class:`JetsonNanoA64`.
    """

    ARCH = 'armv8'


class RaspberryPi4A64(RaspberryPi4):
    """
    Raspberry Pi 4 built around the AArch64 core, matching its Cortex-A72.
    See the caveat on :class:`JetsonNanoA64`.
    """

    ARCH = 'armv8'


__all__ = [
    'Board', 'RaspberryPi3', 'RaspberryPi4', 'JetsonNano',
    'RaspberryPi3A64', 'RaspberryPi4A64', 'JetsonNanoA64', 'JetsonNanoA64Smp',
]
