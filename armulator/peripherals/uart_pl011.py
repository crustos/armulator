"""
ARM PL011 UART -- the Pi's ``ttyAMA0`` and the Jetson's ``ttyTHS*`` lineage.

Transmitted bytes accumulate in :attr:`tx_buffer` so tests can assert on
what the firmware printed; :meth:`feed` pushes bytes into the receive FIFO
as if something were typing at the other end.
"""

from armulator.peripherals.mmio import MMIODevice

# Flag register bits
FR_CTS = 1 << 0
FR_BUSY = 1 << 3
FR_RXFE = 1 << 4    # receive FIFO empty
FR_TXFF = 1 << 5    # transmit FIFO full
FR_RXFF = 1 << 6    # receive FIFO full
FR_TXFE = 1 << 7    # transmit FIFO empty

# Interrupt bits (RIS/MIS/IMSC/ICR)
INT_RX = 1 << 4
INT_TX = 1 << 5


class Pl011Uart(MMIODevice):

    SIZE = 0x1000
    FIFO_DEPTH = 16

    REGISTERS = {
        0x00: 'DR', 0x04: 'RSRECR', 0x18: 'FR', 0x20: 'ILPR',
        0x24: 'IBRD', 0x28: 'FBRD', 0x2C: 'LCRH', 0x30: 'CR',
        0x34: 'IFLS', 0x38: 'IMSC', 0x3C: 'RIS', 0x40: 'MIS',
        0x44: 'ICR', 0x48: 'DMACR',
    }

    def __init__(self, name='uart', trace=False):
        super().__init__(self.SIZE, name=name, trace=trace)
        #: Everything the firmware has transmitted.
        self.tx_buffer = bytearray()
        self._rx_fifo = bytearray()
        self._imsc = 0
        self._ris = 0
        self._cr = 0x0300          # TXE | RXE, matching reset state
        self._lcrh = 0
        self._ibrd = 0
        self._fbrd = 0
        self._ifls = 0x12
        #: Called as fn(byte) for each transmitted byte.
        self.tx_callbacks = []

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------
    def feed(self, data) -> None:
        """Push ``data`` (bytes or str) into the receive FIFO."""
        if isinstance(data, str):
            data = data.encode()
        self._rx_fifo.extend(data)
        self._update_interrupts()

    @property
    def text(self) -> str:
        """Transmitted bytes decoded as UTF-8, errors replaced."""
        return self.tx_buffer.decode('utf-8', errors='replace')

    @property
    def baud_divisors(self):
        """The raw ``(IBRD, FBRD)`` pair the firmware programmed."""
        return self._ibrd, self._fbrd

    def _update_interrupts(self):
        if self._rx_fifo:
            self._ris |= INT_RX
        else:
            self._ris &= ~INT_RX
        self._ris |= INT_TX        # transmit FIFO is always empty here
        self.set_irq(bool(self._ris & self._imsc))

    # ------------------------------------------------------------------
    # Register interface
    # ------------------------------------------------------------------
    def read_register(self, offset):
        if offset == 0x00:                                   # DR
            if not self._rx_fifo:
                return 0
            byte = self._rx_fifo.pop(0)
            self._update_interrupts()
            return byte
        if offset == 0x18:                                   # FR
            flags = FR_TXFE                                  # tx always drains
            if not self._rx_fifo:
                flags |= FR_RXFE
            if len(self._rx_fifo) >= self.FIFO_DEPTH:
                flags |= FR_RXFF
            return flags
        if offset == 0x24:
            return self._ibrd
        if offset == 0x28:
            return self._fbrd
        if offset == 0x2C:
            return self._lcrh
        if offset == 0x30:
            return self._cr
        if offset == 0x34:
            return self._ifls
        if offset == 0x38:
            return self._imsc
        if offset == 0x3C:
            return self._ris
        if offset == 0x40:                                   # MIS
            return self._ris & self._imsc
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        if offset == 0x00:                                   # DR
            byte = value & 0xFF
            self.tx_buffer.append(byte)
            for cb in self.tx_callbacks:
                cb(byte)
            return
        if offset == 0x24:
            self._ibrd = value & 0xFFFF
        elif offset == 0x28:
            self._fbrd = value & 0x3F
        elif offset == 0x2C:
            self._lcrh = value & 0xFF
        elif offset == 0x30:
            self._cr = value & 0xFFFF
        elif offset == 0x34:
            self._ifls = value & 0x3F
        elif offset == 0x38:
            self._imsc = value & 0x7FF
            self._update_interrupts()
        elif offset == 0x44:                                 # ICR -- write 1 to clear
            self._ris &= ~(value & 0x7FF)
            self._update_interrupts()


class BcmSystemTimer(MMIODevice):
    """
    BCM2835 free-running 1 MHz system timer.

    The counter is virtual: it advances via :meth:`tick` rather than wall
    clock, so tests are deterministic and delay loops terminate promptly.
    """

    SIZE = 0x1000

    REGISTERS = {
        0x00: 'CS', 0x04: 'CLO', 0x08: 'CHI',
        0x0C: 'C0', 0x10: 'C1', 0x14: 'C2', 0x18: 'C3',
    }

    def __init__(self, name='systimer', trace=False, auto_advance=1):
        super().__init__(self.SIZE, name=name, trace=trace)
        self.counter = 0
        #: Microseconds added to the counter on every CLO read.  Keeps
        #: firmware busy-wait loops from spinning forever.
        self.auto_advance = auto_advance
        self.compare = [0, 0, 0, 0]
        self._matched = [False] * 4

    def tick(self, microseconds=1) -> None:
        """Advance the virtual counter and latch any compare matches."""
        self.counter = (self.counter + microseconds) & 0xFFFFFFFFFFFFFFFF
        low = self.counter & 0xFFFFFFFF
        for i in range(4):
            if self.compare[i] and low >= self.compare[i]:
                self._matched[i] = True
        self.set_irq(any(self._matched))

    def read_register(self, offset):
        if offset == 0x00:                                   # CS
            return sum(1 << i for i in range(4) if self._matched[i])
        if offset == 0x04:                                   # CLO
            value = self.counter & 0xFFFFFFFF
            if self.auto_advance:
                self.tick(self.auto_advance)
            return value
        if offset == 0x08:                                   # CHI
            return (self.counter >> 32) & 0xFFFFFFFF
        if 0x0C <= offset <= 0x18:
            return self.compare[(offset - 0x0C) // 4]
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        if offset == 0x00:                                   # CS -- write 1 to clear
            for i in range(4):
                if value & (1 << i):
                    self._matched[i] = False
            self.set_irq(any(self._matched))
        elif 0x0C <= offset <= 0x18:
            self.compare[(offset - 0x0C) // 4] = value & 0xFFFFFFFF
