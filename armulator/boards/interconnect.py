"""
Wiring emulated boards to each other.

A single :class:`Board` is a closed world.  This module supplies the wires
between them, so a Raspberry Pi and a Jetson Nano can be co-emulated and
exchange signals the way two real boards on a bench would.

Three pieces:

  :class:`GpioLink`     a wire between a pin on one board and a pin on another
  :class:`SpiBridge`    a Pi SPI master driving a Jetson acting as slave
  :class:`Machine`      runs several boards in lockstep, stepping the wires

The scheduler is cooperative round-robin: each board executes a slice of
instructions, then every link is settled before the next slice.  That is
not cycle-accurate -- these are independent CPUs with independent clocks --
but it is deterministic and repeatable, which is what a test needs.
"""


class GpioLink:
    """
    A wire from an output pin on one board to an input pin on another.

    :param driver: board whose pin drives the wire
    :param driver_pin: pin number (or Tegra pin name) on the driver
    :param receiver: board whose pin is driven
    :param receiver_pin: pin number (or Tegra pin name) on the receiver
    :param inverting: drive the logical complement, for active-low signals

    Only pins actually configured as outputs drive the wire; if the driver
    has its pin as an input the line is released and the receiver falls
    back to its own pull resistor, exactly like an undriven wire.
    """

    def __init__(self, driver, driver_pin, receiver, receiver_pin,
                 inverting=False, name=None):
        self.driver = driver
        self.driver_pin = driver_pin
        self.receiver = receiver
        self.receiver_pin = receiver_pin
        self.inverting = inverting
        self.name = name or f'{driver_pin}->{receiver_pin}'
        #: Every level change seen on this wire, as (step, level).
        self.history = []
        self._last = None

    def _driver_is_output(self):
        gpio = self.driver.gpio
        if hasattr(gpio, 'is_output'):              # Tegra
            return gpio.is_output(self.driver_pin)
        from armulator.peripherals.gpio_bcm import GpioFunction
        return gpio.function(self.driver_pin) == GpioFunction.OUTPUT

    def settle(self, step=0) -> None:
        """Propagate the driver's level to the receiver."""
        if not self._driver_is_output():
            self.receiver.gpio.drive_input(self.receiver_pin, None)
            return
        level = self.driver.gpio.level(self.driver_pin)
        if self.inverting:
            level = not level
        self.receiver.gpio.drive_input(self.receiver_pin, level)
        if level != self._last:
            self.history.append((step, level))
            self._last = level

    def edges(self):
        """Level changes recorded on this wire."""
        return list(self.history)


class SpiBridge:
    """
    Wires one board's SPI master to another board's SPI slave controller.

    Both sides are real peripheral models: the master's FIFO writes drive
    the slave's :meth:`~armulator.peripherals.spi_slave.Bcm2835SpiSlave.transfer`,
    and chip-select framing propagates so the slave sees proper dialogue
    boundaries.  Nothing reaches into either controller's internals.

    :param master: board whose SPI master drives the bus
    :param slave_board: board acting as slave; needs a ``spi_slave`` device
    :param chip_select: which CS line on the master selects this slave

    Because the BCM2835 slave block is half duplex and expects an
    address/direction octet first, the master's firmware must open each
    dialogue with :func:`~armulator.peripherals.spi_slave.address_octet`.
    A master that just shifts raw data will be ignored by the slave -- which
    is exactly what real hardware does, and worth reproducing.
    """

    def __init__(self, master, slave_board, chip_select=0, name='spi_bridge'):
        slave = getattr(slave_board, 'spi_slave', None)
        if slave is None:
            raise ValueError(
                f'{type(slave_board).__name__} has no spi_slave controller; '
                'only boards with a modelled SPI slave can be bridged to'
            )
        self.master = master
        self.slave_board = slave_board
        self.slave = slave
        self.chip_select = chip_select
        self.name = name
        #: (master_byte, slave_byte) for every exchange.
        self.exchanges = []
        master.spi.attach_slave(self, chip_select)

    # The bridge sits between the two controllers so it can record traffic,
    # forwarding every part of the slave contract through to the real model.
    def transfer(self, byte: int) -> int:
        response = self.slave.transfer(byte)
        self.exchanges.append((byte & 0xFF, response & 0xFF))
        return response

    def select(self) -> None:
        self.slave.select()

    def deselect(self) -> None:
        self.slave.deselect()

    def queue_slave_response(self, data) -> None:
        """Load bytes into the slave's TX FIFO for a read dialogue."""
        self.slave.queue_transmit(data)

    @property
    def mosi(self) -> bytes:
        """Everything the master shifted out, including address octets."""
        return bytes(m for m, _ in self.exchanges)

    @property
    def miso(self) -> bytes:
        """Everything the slave drove back."""
        return bytes(s for _, s in self.exchanges)

    def settle(self, step=0) -> None:
        """No continuous state to settle; transfers are event driven."""


class Machine:
    """
    Several boards running together with links between them.

        machine = Machine()
        machine.add('pi', pi)
        machine.add('nano', nano)
        machine.link(GpioLink(pi, 17, nano, 'PA0'))
        machine.run(3000)

    Boards are stepped round-robin in ``slice_size`` instruction chunks,
    with all links settled between slices.  A smaller slice means finer
    interleaving and slower execution; the default trades reasonably.
    """

    def __init__(self, slice_size=16):
        self.boards = {}
        self.links = []
        self.slice_size = slice_size
        self.steps = 0

    # ------------------------------------------------------------------
    def add(self, name, board):
        self.boards[name] = board
        setattr(self, name, board)
        return board

    def link(self, link):
        self.links.append(link)
        return link

    def settle(self) -> None:
        """Propagate every wire once."""
        for wire in self.links:
            wire.settle(self.steps)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Settle the initial state before any instruction executes."""
        self.settle()

    def step_slice(self) -> int:
        """
        Run one round: a slice of instructions on each board, then settle.

        Returns the total instructions executed this round.
        """
        executed = 0
        for board in self.boards.values():
            executed += board.run(self.slice_size)
        self.steps += 1
        self.settle()
        return executed

    @property
    def all_halted(self) -> bool:
        """
        True when every board has parked on its halt loop.

        A halted board still reports a couple of executed instructions per
        slice while it re-detects the spin, so this flag -- not the
        instruction count -- is what tells the scheduler to stop.
        """
        return bool(self.boards) and all(b.halted for b in self.boards.values())

    def run(self, max_instructions=10000) -> int:
        """
        Run all boards until they stop advancing or the budget is spent.

        Returns total instructions executed.  Stops early once every board
        has parked (all firmware sitting on its halt loop), so tests do not
        burn their whole budget spinning.
        """
        total = 0
        budget = max_instructions
        while budget > 0:
            executed = self.step_slice()
            total += executed
            budget -= max(executed, 1)
            if self.all_halted:
                break
        return total

    def run_until(self, predicate, max_instructions=10000) -> bool:
        """
        Run until ``predicate()`` is true or the budget is spent.

        Returns whether the predicate was satisfied -- useful for waiting
        on a signal to cross between boards without guessing a step count.
        """
        budget = max_instructions
        self.settle()
        if predicate():
            return True
        while budget > 0:
            executed = self.step_slice()
            if predicate():
                return True
            budget -= max(executed, 1)
            if self.all_halted:
                break
        return False
