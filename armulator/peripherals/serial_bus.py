"""
Serial bus controllers: BCM2835 SPI0 and BSC (I2C).

Both models are built around a *bus* object rather than a hardcoded
peripheral, so a controller can be pointed at a Python-implemented slave, a
loopback, or -- the interesting case -- a controller on a *different
emulated board*.  That is what makes cross-device examples possible: a Pi
acting as SPI master to a Jetson acting as slave.

Slave devices implement one method::

    class MySlave:
        def transfer(self, byte: int) -> int:
            '''Receive a byte from the master, return the byte shifted back.'''

I2C slaves additionally carry an ``address`` attribute and implement
``read(count)`` / ``write(data)``.
"""

from armulator.peripherals.mmio import MMIODevice

# ----------------------------------------------------------------------
# SPI -- BCM2835 SPI0
# ----------------------------------------------------------------------
SPI_CS = 0x00
SPI_FIFO = 0x04
SPI_CLK = 0x08
SPI_DLEN = 0x0C

# CS register bits
CS_CHIP_SELECT = 0b11
CS_CLEAR_TX = 1 << 4
CS_CLEAR_RX = 1 << 5
CS_CPOL = 1 << 3
CS_CPHA = 1 << 2
CS_TA = 1 << 7           # transfer active
CS_DONE = 1 << 16
CS_RXD = 1 << 17         # RX FIFO contains data
CS_TXD = 1 << 18         # TX FIFO accepts data
CS_RXR = 1 << 19
CS_RXF = 1 << 20
CS_INTR = 1 << 6
CS_INTD = 1 << 9


class SpiLoopback:
    """Trivial slave that echoes whatever it receives."""

    def transfer(self, byte):
        return byte


class SpiSlaveDevice:
    """
    A recording SPI slave.

    Returns bytes from :attr:`responses` in order (0x00 once exhausted) and
    logs everything the master sent to :attr:`received`.
    """

    def __init__(self, responses=None, name='spi_slave'):
        self.name = name
        self.received = bytearray()
        self.responses = bytearray(responses or b'')
        self._index = 0

    def transfer(self, byte):
        self.received.append(byte)
        if self._index < len(self.responses):
            out = self.responses[self._index]
            self._index += 1
            return out
        return 0x00

    def queue(self, data):
        """Add bytes for the slave to return on subsequent transfers."""
        self.responses.extend(data if isinstance(data, (bytes, bytearray))
                              else bytes(data))


class Bcm2835Spi(MMIODevice):
    """
    BCM2835 SPI0 master.

    Each byte written to :data:`SPI_FIFO` is handed to the slave selected by
    the CS field immediately, and the byte it returns is pushed into the
    receive FIFO -- SPI is full duplex, so one write always yields one read.
    """

    SIZE = 0x1000

    REGISTERS = {
        SPI_CS: 'CS', SPI_FIFO: 'FIFO', SPI_CLK: 'CLK', SPI_DLEN: 'DLEN',
    }

    def __init__(self, name='spi', trace=False):
        super().__init__(self.SIZE, name=name, trace=trace)
        self._cs = 0
        self._clk = 0
        self._dlen = 0
        self._rx = bytearray()
        #: Chip select line -> slave object.
        self.slaves = {}
        #: Everything the master has shifted out, in order.
        self.transmitted = bytearray()

    # ------------------------------------------------------------------
    def attach_slave(self, slave, chip_select=0):
        """Wire ``slave`` to chip select line ``chip_select`` (0-2)."""
        self.slaves[chip_select] = slave
        return slave

    @property
    def chip_select(self):
        return self._cs & CS_CHIP_SELECT

    @property
    def transfer_active(self):
        return bool(self._cs & CS_TA)

    @property
    def clock_divider(self):
        return self._clk

    def _exchange(self, byte):
        slave = self.slaves.get(self.chip_select)
        self.transmitted.append(byte)
        received = slave.transfer(byte) if slave is not None else 0x00
        self._rx.append(received & 0xFF)
        self._update_interrupt()

    def _update_interrupt(self):
        # INTR raises on RXR, INTD on DONE; both are edge-ish in practice.
        want = False
        if self._cs & CS_INTR and self._rx:
            want = True
        if self._cs & CS_INTD and not self.transfer_active:
            want = True
        self.set_irq(want)

    # ------------------------------------------------------------------
    def read_register(self, offset):
        if offset == SPI_CS:
            status = self._cs & ~(CS_DONE | CS_RXD | CS_TXD | CS_RXR | CS_RXF)
            status |= CS_TXD                      # TX FIFO never fills here
            if self._rx:
                status |= CS_RXD | CS_RXR
            if not self.transfer_active:
                status |= CS_DONE
            return status
        if offset == SPI_FIFO:
            if not self._rx:
                return 0
            byte = self._rx.pop(0)
            self._update_interrupt()
            return byte
        if offset == SPI_CLK:
            return self._clk
        if offset == SPI_DLEN:
            return self._dlen
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        if offset == SPI_CS:
            if value & CS_CLEAR_RX:
                self._rx.clear()
            # The clear bits are write-only strobes, not state.
            self._cs = value & ~(CS_CLEAR_TX | CS_CLEAR_RX)
            self._update_interrupt()
        elif offset == SPI_FIFO:
            self._exchange(value & 0xFF)
        elif offset == SPI_CLK:
            self._clk = value & 0xFFFF
        elif offset == SPI_DLEN:
            self._dlen = value & 0xFFFF


# ----------------------------------------------------------------------
# I2C -- BCM2835 BSC
# ----------------------------------------------------------------------
BSC_C = 0x00
BSC_S = 0x04
BSC_DLEN = 0x08
BSC_A = 0x0C
BSC_FIFO = 0x10
BSC_DIV = 0x14

# Control register bits
C_READ = 1 << 0
C_CLEAR = 0b11 << 4
C_ST = 1 << 7            # start transfer
C_INTD = 1 << 8
C_INTT = 1 << 9
C_INTR = 1 << 10
C_I2CEN = 1 << 15

# Status register bits
S_TA = 1 << 0            # transfer active
S_DONE = 1 << 1
S_TXW = 1 << 2
S_RXR = 1 << 3
S_TXD = 1 << 4
S_RXD = 1 << 5
S_TXE = 1 << 6
S_RXF = 1 << 7
S_ERR = 1 << 8           # slave did not acknowledge
S_CLKT = 1 << 9


class I2cSlaveDevice:
    """
    A simple register-file I2C slave.

    Writes of the form ``[reg, value...]`` store into :attr:`registers`;
    reads return from the last addressed register onward.
    """

    def __init__(self, address, registers=None, name='i2c_slave'):
        self.address = address
        self.name = name
        self.registers = dict(registers or {})
        self.received = []
        self._pointer = 0

    def write(self, data):
        self.received.append(bytes(data))
        if not data:
            return
        self._pointer = data[0]
        for i, byte in enumerate(data[1:]):
            self.registers[self._pointer + i] = byte

    def read(self, count):
        out = bytearray()
        for i in range(count):
            out.append(self.registers.get(self._pointer + i, 0xFF) & 0xFF)
        return bytes(out)


class Bcm2835I2c(MMIODevice):
    """
    BCM2835 BSC (Broadcom Serial Controller) I2C master.

    Transfers are transactional: firmware sets the slave address, the byte
    count in DLEN, then sets ST in the control register.  A read pulls DLEN
    bytes from the slave into the FIFO; a write drains the FIFO to the
    slave.  If no slave answers the address, ERR is set in the status
    register -- the NACK path real drivers must handle.
    """

    SIZE = 0x1000

    REGISTERS = {
        BSC_C: 'C', BSC_S: 'S', BSC_DLEN: 'DLEN',
        BSC_A: 'A', BSC_FIFO: 'FIFO', BSC_DIV: 'DIV',
    }

    def __init__(self, name='i2c', trace=False):
        super().__init__(self.SIZE, name=name, trace=trace)
        self._control = 0
        self._dlen = 0
        self._address = 0
        self._div = 0
        self._tx = bytearray()
        self._rx = bytearray()
        self._done = False
        self._error = False
        #: 7-bit address -> slave object.
        self.slaves = {}

    # ------------------------------------------------------------------
    def attach_slave(self, slave, address=None):
        """Put ``slave`` on the bus at its own address, or an override."""
        self.slaves[address if address is not None else slave.address] = slave
        return slave

    @property
    def target_address(self):
        return self._address

    @property
    def clock_divider(self):
        return self._div

    def _start_transfer(self):
        slave = self.slaves.get(self._address)
        self._done = False
        self._error = False
        if slave is None:
            self._error = True                    # NACK: nobody home
            self._done = True
            self._update_interrupt()
            return
        if self._control & C_READ:
            self._rx.extend(slave.read(self._dlen))
        else:
            payload = bytes(self._tx[:self._dlen])
            self._tx = self._tx[len(payload):]
            slave.write(payload)
        self._done = True
        self._update_interrupt()

    def _update_interrupt(self):
        want = False
        if self._control & C_INTD and self._done:
            want = True
        if self._control & C_INTR and self._rx:
            want = True
        self.set_irq(want)

    # ------------------------------------------------------------------
    def read_register(self, offset):
        if offset == BSC_C:
            return self._control & ~C_ST          # ST is a write-only strobe
        if offset == BSC_S:
            status = 0
            if self._done:
                status |= S_DONE
            if self._error:
                status |= S_ERR
            if self._rx:
                status |= S_RXD | S_RXR
            else:
                status |= S_RXF & 0                # RX empty
            if not self._tx:
                status |= S_TXE
            status |= S_TXD                        # FIFO always accepts
            return status
        if offset == BSC_DLEN:
            return self._dlen
        if offset == BSC_A:
            return self._address
        if offset == BSC_FIFO:
            if not self._rx:
                return 0
            byte = self._rx.pop(0)
            self._update_interrupt()
            return byte
        if offset == BSC_DIV:
            return self._div
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        if offset == BSC_C:
            if value & C_CLEAR:
                self._tx.clear()
            self._control = value & ~C_CLEAR
            if value & C_ST:
                self._start_transfer()
            else:
                self._update_interrupt()
        elif offset == BSC_S:
            if value & S_DONE:
                self._done = False
            if value & S_ERR:
                self._error = False
            self._update_interrupt()
        elif offset == BSC_DLEN:
            self._dlen = value & 0xFFFF
        elif offset == BSC_A:
            self._address = value & 0x7F
        elif offset == BSC_FIFO:
            self._tx.append(value & 0xFF)
        elif offset == BSC_DIV:
            self._div = value & 0xFFFF
