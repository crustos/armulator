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
armulator's CPU core is ARMv6 (A32/T32, integer only).  The Pi 3, Pi 4 and
Jetson Nano all use ARMv8-A cores, so these boards will *not* execute
AArch64 binaries or the stock vendor kernels.  What they do model faithfully
is the peripheral register interface, which is where GPIO driver logic
actually lives.  Write your test firmware as 32-bit ARM (``-marm
-march=armv6``) and it will exercise exactly the same register sequences
your AArch64 driver performs.
"""

from armulator.armv6.arm_v6 import ArmV6
from armulator.armv6.enums import InstrSet
from armulator.armv6.memory_controller_hub import MemoryController
from armulator.armv6.memory_types import RAM
from armulator.peripherals.gpio_bcm import BcmGpio
from armulator.peripherals.gic400 import SPI_BASE, Gic400
from armulator.peripherals.gpio_tegra import TegraGpio
from armulator.peripherals.serial_bus import Bcm2835I2c, Bcm2835Spi
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

    def __init__(self, ram_size=0x100000, trace=False):
        self.cpu = ArmV6()
        self.trace = trace
        self.devices = {}
        #: Interrupt controller, if this board has one.
        self.gic = None
        #: True once firmware has parked on its halt loop.
        self.halted = False

        # Flat physical addressing: the MPU/MMU is off, so firmware sees
        # the peripheral map directly, as it does at reset on real hardware.
        self.cpu.registers.sctlr.m = 0

        self.ram = RAM(ram_size)
        self.cpu.mem.memories = [
            MemoryController(self.ram, self.RAM_BASE, self.RAM_BASE + ram_size)
        ]
        self._build()
        self.cpu.take_reset()

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
        self.cpu.mem.memories.append(
            MemoryController(device, base, base + device.size)
        )
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

        The instruction set is selected explicitly rather than inherited
        from SCTLR.TE, which resets to Thumb in armulator's default
        configuration and would silently misdecode A32 firmware.
        """
        self.halted = False
        self.cpu.registers.select_instr_set(InstrSet.THUMB if thumb else InstrSet.ARM)
        self.cpu.registers.branch_to(self.CODE_BASE if address is None else address)

    def step(self) -> None:
        """Execute one instruction and deliver any pending peripheral IRQ."""
        self.cpu.emulate_cycle()
        self.service_interrupts()

    def run(self, max_instructions=10000, stop_at=None) -> int:
        """
        Run until ``stop_at`` is reached or ``max_instructions`` execute.

        Returns the number of instructions actually executed.  Also stops
        early if the CPU parks on a tight self-branch (``B .``), the usual
        way bare-metal test firmware signals that it is done.
        """
        executed = 0
        previous_pc = None
        for _ in range(max_instructions):
            pc = self.cpu.registers.pc_store_value()
            if stop_at is not None and pc == stop_at:
                break
            self.step()
            executed += 1
            new_pc = self.cpu.registers.pc_store_value()
            if new_pc == pc and pc == previous_pc:
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
            self.gic.refresh()
            asserting = self.gic.irq_pending
        else:
            asserting = bool(self.pending_irq())
        if self.cpu.registers.cpsr.i:
            return False
        if not asserting:
            return False
        self.cpu.registers.take_physical_irq_exception()
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

    def _build(self):
        self.attach('gpio', BcmGpio(pull_style='legacy', name='gpio'),
                    offset=self.GPIO_OFFSET)
        self.attach('uart', Pl011Uart(name='uart0'), offset=self.UART0_OFFSET)
        self.attach('timer', BcmSystemTimer(name='systimer'),
                    offset=self.SYSTIMER_OFFSET)
        self.attach('spi', Bcm2835Spi(name='spi0'), offset=self.SPI0_OFFSET)
        self.attach('i2c', Bcm2835I2c(name='i2c1'), offset=self.I2C1_OFFSET)


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
    Only the GPIO block and a debug UART are modelled -- the Tegra boot ROM
    and CBoot chain are proprietary and out of scope.
    """

    PERIPHERAL_BASE = 0x60000000
    CODE_BASE = 0x80080000
    RAM_BASE = 0x80000000

    GPIO_ADDRESS = 0x6000D000
    UARTA_ADDRESS = 0x70006000
    SPI_ADDRESS = 0x7000D400

    #: Tegra X1's GIC-500 distributor base.  GICv2-compatible registers.
    GIC_ADDRESS = 0x50041000

    GPIO_SPI = 32
    UART_SPI = 36
    SPI_SPI = 39

    def _build(self):
        self.attach('gpio', TegraGpio(name='tegra_gpio'), address=self.GPIO_ADDRESS)
        self.attach('uart', Pl011Uart(name='uarta'), address=self.UARTA_ADDRESS)
        self.attach('spi', Bcm2835Spi(name='spi1'), address=self.SPI_ADDRESS)

        self.gic = self.attach('gic', Gic400(name='gic500'),
                               address=self.GIC_ADDRESS)
        self.connect_irq(self.gpio, self.GPIO_SPI)
        self.connect_irq(self.uart, self.UART_SPI)
        self.connect_irq(self.spi, self.SPI_SPI)
