"""
BCM2835 SPI/BSC slave controller.

This is the peripheral that lets a Raspberry Pi be the *slave* on an SPI
bus (BCM2835 ARM Peripherals ch. 11, base 0x7E214000).  It is a genuinely
odd block, and modelling it faithfully is the point -- a driver written
against an idealised full-duplex slave will not work on real silicon.

THE DIALOGUE PROTOCOL
---------------------
The block does not do plain full-duplex SPI.  Transfers are octet-based
"dialogues", MSB first, and the *first* MOSI octet of each chip-select
assertion is an address/direction byte rather than data:

    first octet LSB == 0   write dialogue: every subsequent MOSI octet is
                           deserialised into the RX FIFO, and nothing is
                           driven on MISO (the line idles high)
    first octet LSB == 1   read dialogue: subsequent MOSI octets are
                           discarded, and the TX FIFO is serialised out
                           on MISO instead

The upper 7 bits of that first octet are the slave address, compared
against the SLV register -- I2C addressing semantics carried over to the
SPI side of a shared block, which is why it looks so unlike normal SPI.

So a dialogue is half duplex.  You cannot send and receive in the same
transaction the way a conventional SPI slave allows.

ERRATA MODELLED
---------------
The BCM2835 datasheet for this block is thin and partly wrong; the
behaviours below are drawn from community reverse engineering of real
hardware rather than the datasheet:

  * CR.BRK does not actually clear the FIFOs.  Firmware that relies on it
    to discard stale TX data will silently transmit that stale data.
    :attr:`brk_clears_fifos` defaults to ``False`` to reproduce this; set
    it ``True`` to model the behaviour the datasheet *describes*, which is
    useful for showing a driver depends on a bug.
  * There is no separate FIFO register.  The low byte of DR is both the RX
    FIFO read port and the TX FIFO write port.
  * MISO idles high (0xFF), not low, during a write dialogue.

UNCERTAIN
---------
The datasheet's interrupt bit assignments for this block are ambiguous and
could not be confirmed against hardware here.  :data:`INT_RX` and
:data:`INT_TX` use bits 0 and 1; if you are testing a driver that depends
on the exact IMSC/RIS layout, verify those two constants against silicon
before trusting a passing test.
"""

from armulator.peripherals.mmio import MMIODevice

# Register offsets (confirmed against the BCM2835 peripherals datasheet)
SLV_DR = 0x00
SLV_RSR = 0x04
SLV_SLV = 0x08
SLV_CR = 0x0C
SLV_FR = 0x10
SLV_IFLS = 0x14
SLV_IMSC = 0x18
SLV_RIS = 0x1C
SLV_MIS = 0x20
SLV_ICR = 0x24
SLV_DMACR = 0x28
SLV_TDR = 0x2C
SLV_GPUSTAT = 0x30
SLV_HCTRL = 0x34

# Control register bits
CR_EN = 1 << 0           # enable the block
CR_SPI = 1 << 1          # select SPI mode
CR_I2C = 1 << 2          # select I2C mode
CR_CPHA = 1 << 3
CR_CPOL = 1 << 4
CR_ENSTAT = 1 << 5
CR_ENCTRL = 1 << 6
CR_BRK = 1 << 7          # "clear FIFOs" -- does not work on real hardware
CR_TXE = 1 << 8          # transmit enable
CR_RXE = 1 << 9          # receive enable
CR_INV_RXF = 1 << 10
CR_TESTFIFO = 1 << 11
CR_HOSTCTRLEN = 1 << 12
CR_INV_TXF = 1 << 13

# Flag register bits
FR_TXFE = 1 << 0         # transmit FIFO empty
FR_RXFF = 1 << 1         # receive FIFO full
FR_TXFF = 1 << 2         # transmit FIFO full
FR_RXFE = 1 << 3         # receive FIFO empty
FR_TXBUSY = 1 << 4
FR_RXBUSY = 1 << 5

# Interrupt bits -- see the UNCERTAIN note in the module docstring.
INT_RX = 1 << 0
INT_TX = 1 << 1

# Receive status register bits
RSR_OE = 1 << 0          # overrun: a byte arrived with the RX FIFO full

#: Level MISO idles at when the slave is not driving it.
MISO_IDLE = 0xFF


class Bcm2835SpiSlave(MMIODevice):
    """
    A Pi acting as an SPI slave.

    Attaches to a master with ``master.spi.attach_slave(slave)``: this class
    implements the :meth:`transfer` contract the master's FIFO writes drive,
    plus :meth:`select` / :meth:`deselect` so chip-select framing starts a
    new dialogue.

    :param fifo_depth: FIFO depth in bytes.  The real depth is undocumented;
        16 is the usual community estimate.
    :param brk_clears_fifos: model CR.BRK as working (it does not on real
        hardware -- see the module docstring).
    """

    SIZE = 0x1000
    FIFO_DEPTH = 16

    REGISTERS = {
        SLV_DR: 'DR', SLV_RSR: 'RSR', SLV_SLV: 'SLV', SLV_CR: 'CR',
        SLV_FR: 'FR', SLV_IFLS: 'IFLS', SLV_IMSC: 'IMSC', SLV_RIS: 'RIS',
        SLV_MIS: 'MIS', SLV_ICR: 'ICR', SLV_DMACR: 'DMACR', SLV_TDR: 'TDR',
        SLV_GPUSTAT: 'GPUSTAT', SLV_HCTRL: 'HCTRL',
    }

    def __init__(self, name='spi_slave', trace=False, fifo_depth=FIFO_DEPTH,
                 brk_clears_fifos=False):
        super().__init__(self.SIZE, name=name, trace=trace)
        self.fifo_depth = fifo_depth
        self.brk_clears_fifos = brk_clears_fifos

        self._cr = 0
        self._slv = 0
        self._imsc = 0
        self._ifls = 0
        self._overrun = False
        self._rx = bytearray()
        self._tx = bytearray()

        # Dialogue state, reset on every chip-select assertion.
        self._selected = False
        self._awaiting_address = True
        self._reading = False           # True once a read dialogue is under way
        self._addressed = False         # True if the address matched SLV

        #: Every dialogue seen, as (address, is_read, payload bytes).
        self.dialogues = []
        self._current = bytearray()

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------
    @property
    def enabled(self):
        return bool(self._cr & CR_EN)

    @property
    def spi_mode(self):
        return bool(self._cr & CR_SPI)

    @property
    def address(self):
        """The slave address firmware programmed into SLV."""
        return self._slv

    @property
    def received(self):
        """Bytes currently sitting in the RX FIFO."""
        return bytes(self._rx)

    @property
    def pending_transmit(self):
        """Bytes currently queued in the TX FIFO."""
        return bytes(self._tx)

    def queue_transmit(self, data):
        """
        Push bytes into the TX FIFO directly (bypassing DR writes).

        Convenient for setting up a test; firmware would write DR instead.
        """
        if isinstance(data, str):
            data = data.encode()
        for byte in data:
            if len(self._tx) < self.fifo_depth:
                self._tx.append(byte)
        self._update_interrupts()

    # ------------------------------------------------------------------
    # Bus-facing interface (called by the master's controller)
    # ------------------------------------------------------------------
    def select(self) -> None:
        """Chip select asserted -- a new dialogue begins."""
        self._selected = True
        self._awaiting_address = True
        self._reading = False
        self._addressed = False
        self._current = bytearray()

    def deselect(self) -> None:
        """Chip select deasserted -- the dialogue ends."""
        if self._selected and not self._awaiting_address:
            self.dialogues.append(
                (self._slv if self._addressed else None,
                 self._reading, bytes(self._current))
            )
        self._selected = False
        self._awaiting_address = True
        self._current = bytearray()

    def transfer(self, byte: int) -> int:
        """
        Exchange one octet with the master.

        Returns what the slave drives on MISO for this octet, honouring the
        half-duplex dialogue rules described in the module docstring.
        """
        byte &= 0xFF
        if not (self.enabled and self.spi_mode):
            return MISO_IDLE
        if not self._selected:
            # Some masters never touch TA; treat the first byte as an
            # implicit select so such firmware still works.
            self.select()

        if self._awaiting_address:
            self._awaiting_address = False
            self._reading = bool(byte & 0x01)
            self._addressed = (byte >> 1) == self._slv
            return MISO_IDLE               # address octet drives nothing back

        if not self._addressed:
            return MISO_IDLE               # dialogue is not for us

        if self._reading:
            # Read dialogue: MOSI is discarded, TX FIFO goes out on MISO.
            self._current.append(byte)
            if not (self._cr & CR_TXE) or not self._tx:
                self._update_interrupts()
                return MISO_IDLE
            out = self._tx.pop(0)
            self._update_interrupts()
            return out

        # Write dialogue: MOSI lands in the RX FIFO, MISO idles.
        self._current.append(byte)
        if self._cr & CR_RXE:
            if len(self._rx) < self.fifo_depth:
                self._rx.append(byte)
            else:
                self._overrun = True       # RX FIFO full: byte is lost
        self._update_interrupts()
        return MISO_IDLE

    # ------------------------------------------------------------------
    def _update_interrupts(self):
        ris = 0
        if self._rx:
            ris |= INT_RX
        if not self._tx:
            ris |= INT_TX
        self._ris = ris
        self.set_irq(bool(ris & self._imsc))

    # ------------------------------------------------------------------
    # Register interface
    # ------------------------------------------------------------------
    def read_register(self, offset):
        if offset == SLV_DR:
            if not self._rx:
                return 0
            byte = self._rx.pop(0)
            self._update_interrupts()
            return byte
        if offset == SLV_RSR:
            return RSR_OE if self._overrun else 0
        if offset == SLV_SLV:
            return self._slv
        if offset == SLV_CR:
            # BRK is a strobe, not stored state.
            return self._cr & ~CR_BRK
        if offset == SLV_FR:
            flags = 0
            if not self._tx:
                flags |= FR_TXFE
            if len(self._tx) >= self.fifo_depth:
                flags |= FR_TXFF
            if not self._rx:
                flags |= FR_RXFE
            if len(self._rx) >= self.fifo_depth:
                flags |= FR_RXFF
            if self._selected:
                flags |= FR_RXBUSY if not self._reading else FR_TXBUSY
            return flags
        if offset == SLV_IFLS:
            return self._ifls
        if offset == SLV_IMSC:
            return self._imsc
        if offset == SLV_RIS:
            self._update_interrupts()
            return self._ris
        if offset == SLV_MIS:
            self._update_interrupts()
            return self._ris & self._imsc
        if offset == SLV_TDR:
            # Test FIFO port: peeks the top TX entry without removing it,
            # which is exactly why it cannot be used to drain the FIFO.
            if self._cr & CR_TESTFIFO and self._tx:
                return self._tx[0]
            return 0
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        if offset == SLV_DR:
            if len(self._tx) < self.fifo_depth:
                self._tx.append(value & 0xFF)
            self._update_interrupts()
        elif offset == SLV_RSR:
            if value & RSR_OE:
                self._overrun = False
        elif offset == SLV_SLV:
            self._slv = value & 0x7F
        elif offset == SLV_CR:
            if value & CR_BRK and self.brk_clears_fifos:
                # Only clears when the caller has opted into the datasheet's
                # described behaviour; real hardware ignores BRK entirely.
                self._tx.clear()
                self._rx.clear()
            self._cr = value & ~CR_BRK
            self._update_interrupts()
        elif offset == SLV_IFLS:
            self._ifls = value & 0x3F
        elif offset == SLV_IMSC:
            self._imsc = value & 0x3
            self._update_interrupts()
        elif offset == SLV_ICR:
            if value & RSR_OE:
                self._overrun = False
            self._update_interrupts()


def address_octet(address: int, read: bool) -> int:
    """
    Build the first octet of a dialogue.

    Handy in test firmware and assertions, since getting this byte wrong is
    the most common way to talk to this peripheral incorrectly.
    """
    if not 0 <= address <= 0x7F:
        raise ValueError('slave address must be 7-bit')
    return ((address & 0x7F) << 1) | (1 if read else 0)
