"""
NS16550-style UART -- the Tegra X1's ``UART-A``, and the 8250 lineage generally.

This is not a variation on the PL011 in :mod:`armulator.peripherals.uart_pl011`;
the two are unrelated designs, and firmware written for one is silently wrong on
the other:

======================  ==========================  ==========================
\\                       PL011                       16550 / Tegra
======================  ==========================  ==========================
data register           ``DR`` at 0x00              ``THR`` at 0x00
"ready to write?"       ``FR.TXFF`` *set* when      ``LSR.THRE`` *set* when
                        the FIFO is full            the register is empty
baud rate               ``IBRD``/``FBRD``, plain    ``DLL``/``DLM``, reachable
                        registers                   only while ``LCR.DLAB``
register spacing        4 bytes                     1 byte architecturally;
                                                    Tegra spaces them 4 apart
======================  ==========================  ==========================

The polarity difference is the dangerous one: a driver that copies the PL011
wait loop shape waits exactly when it should write. The spacing is the other
trap -- with the unshifted offsets, ``LCR`` writes land on ``IIR``/``FCR`` and
the port is configured almost at random, with no error, because every address
in the range decodes to *some* real register.

Both traps are modelled rather than smoothed over. A driver that gets the
spacing wrong will scribble on the wrong registers here too, and one that
inverts the ``THRE`` poll will hang -- which is the point: the failure should
reproduce in the emulator instead of waiting for hardware.

Transmitted bytes accumulate in :attr:`tx_buffer` so tests can assert on what
the firmware printed; :meth:`feed` pushes bytes into the receive FIFO as if
something were typing at the other end. Both mirror :class:`Pl011Uart`, so a
test can swap one console for the other.
"""

from armulator.peripherals.mmio import MMIODevice

# Register indices, *before* the address shift is applied.
REG_RBR = 0     # read:  received byte
REG_THR = 0     # write: byte to transmit
REG_DLL = 0     # divisor low,  when LCR.DLAB is set
REG_IER = 1
REG_DLM = 1     # divisor high, when LCR.DLAB is set
REG_IIR = 2     # read
REG_FCR = 2     # write
REG_LCR = 3
REG_MCR = 4
REG_LSR = 5
REG_MSR = 6
REG_SCR = 7     # scratch

# Line control
LCR_WLEN8 = 0x03
LCR_DLAB = 0x80

# Interrupt enable
IER_ERBFI = 0x01    # received data available
IER_ETBEI = 0x02    # transmit holding register empty

# Interrupt identification (read from IIR)
IIR_NONE = 0x01     # bit 0 *set* means no interrupt is pending
IIR_THRE = 0x02
IIR_RDA = 0x04
IIR_FIFO_EN = 0xC0

# FIFO control
FCR_ENABLE = 0x01
FCR_CLR_RX = 0x02
FCR_CLR_TX = 0x04

# Modem control
MCR_DTR = 0x01
MCR_RTS = 0x02

# Line status
LSR_DR = 0x01       # a byte has been received
LSR_THRE = 0x20     # transmit holding register empty -- ready
LSR_TEMT = 0x40     # transmitter completely idle


class Uart8250(MMIODevice):
    """
    A 16550 UART with configurable register spacing.

    :param shift: log2 of the distance between registers. Tegra uses ``2``
        (4-byte spacing); a plain ISA 16550 uses ``0``.
    :param size: bytes of address space to occupy. The default 0x40 matches
        the Tegra window, where UART-B follows at +0x40.

    Unlike most devices here, accesses are decoded per *byte* rather than per
    32-bit word, because a 16550's registers are 8 bits wide however far apart
    they are spaced. With ``shift=2`` the three padding bytes after each
    register read as zero and ignore writes, which is what a 32-bit access to
    a byte-wide register on a 32-bit bus sees.
    """

    SIZE = 0x40
    FIFO_DEPTH = 16

    NAMES = {
        REG_THR: 'THR', REG_IER: 'IER', REG_IIR: 'IIR', REG_LCR: 'LCR',
        REG_MCR: 'MCR', REG_LSR: 'LSR', REG_MSR: 'MSR', REG_SCR: 'SCR',
    }
    #: Names used instead of the above while ``LCR.DLAB`` is set.
    DLAB_NAMES = {REG_DLL: 'DLL', REG_DLM: 'DLM'}

    def __init__(self, name='uart8250', shift=2, size=None, trace=False):
        super().__init__(self.SIZE if size is None else size,
                         name=name, trace=trace)
        self.shift = shift
        #: Everything the firmware has transmitted.
        self.tx_buffer = bytearray()
        self._rx_fifo = bytearray()
        self._ier = 0
        self._lcr = 0
        self._mcr = 0
        self._fcr = 0
        self._scr = 0
        self._dll = 0
        self._dlm = 0
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
    def divisor(self) -> int:
        """The divisor the firmware programmed into ``DLL``/``DLM``."""
        return self._dll | (self._dlm << 8)

    @property
    def dlab(self) -> bool:
        """True while the divisor latches are selected."""
        return bool(self._lcr & LCR_DLAB)

    @property
    def fifos_enabled(self) -> bool:
        return bool(self._fcr & FCR_ENABLE)

    def baud(self, clock: int) -> float:
        """Baud rate implied by the programmed divisor at ``clock`` Hz."""
        divisor = self.divisor
        if divisor == 0:
            raise ValueError('no divisor programmed')
        return clock / (16.0 * divisor)

    # ------------------------------------------------------------------
    # Interrupts
    # ------------------------------------------------------------------
    def _update_interrupts(self):
        pending = False
        if (self._ier & IER_ERBFI) and self._rx_fifo:
            pending = True
        if self._ier & IER_ETBEI:
            # The transmitter drains instantly here, so THRE is always true.
            pending = True
        self.set_irq(pending)

    def _iir(self):
        # Receive-data-available outranks transmitter-empty, as on hardware.
        if (self._ier & IER_ERBFI) and self._rx_fifo:
            value = IIR_RDA
        elif self._ier & IER_ETBEI:
            value = IIR_THRE
        else:
            value = IIR_NONE
        if self.fifos_enabled:
            value |= IIR_FIFO_EN
        return value

    # ------------------------------------------------------------------
    # Register interface, in 16550 register *indices*
    # ------------------------------------------------------------------
    def read_index(self, index: int) -> int:
        if index == REG_RBR:
            if self.dlab:
                return self._dll
            if not self._rx_fifo:
                return 0
            byte = self._rx_fifo.pop(0)
            self._update_interrupts()
            return byte
        if index == REG_IER:
            return self._dlm if self.dlab else self._ier
        if index == REG_IIR:
            return self._iir()
        if index == REG_LCR:
            return self._lcr
        if index == REG_MCR:
            return self._mcr
        if index == REG_LSR:
            # The transmitter never backs up in this model, so THRE and TEMT
            # are always set; a driver polling either makes progress.
            status = LSR_THRE | LSR_TEMT
            if self._rx_fifo:
                status |= LSR_DR
            return status
        if index == REG_MSR:
            # CTS/DSR/DCD asserted, so flow-controlled drivers proceed.
            return 0xB0
        if index == REG_SCR:
            return self._scr
        return self.DEFAULT_READ

    def write_index(self, index: int, value: int) -> None:
        value &= 0xFF
        if index == REG_THR:
            if self.dlab:
                self._dll = value
                return
            self.tx_buffer.append(value)
            for cb in self.tx_callbacks:
                cb(value)
            self._update_interrupts()
            return
        if index == REG_IER:
            if self.dlab:
                self._dlm = value
            else:
                self._ier = value
                self._update_interrupts()
            return
        if index == REG_FCR:
            self._fcr = value
            if value & FCR_CLR_RX:
                self._rx_fifo.clear()
            self._update_interrupts()
            return
        if index == REG_LCR:
            self._lcr = value
            return
        if index == REG_MCR:
            self._mcr = value
            return
        if index == REG_SCR:
            self._scr = value
            return
        # LSR and MSR are read-only; writes are discarded as on hardware.

    # ------------------------------------------------------------------
    # Byte-granular decode
    # ------------------------------------------------------------------
    def _index_for(self, address):
        """Register index at byte ``address``, or None if it is padding."""
        stride = 1 << self.shift
        if address & (stride - 1):
            return None
        return address >> self.shift

    def register_name(self, offset):
        index = self._index_for(offset)
        if index is None:
            return f'+0x{offset:03X}'
        if self.dlab and index in self.DLAB_NAMES:
            return self.DLAB_NAMES[index]
        return self.NAMES.get(index, f'reg{index}')

    # MMIODevice's word-based path assumes 32-bit registers, so both accessors
    # are replaced. Tracing still goes through _record, so device traces and
    # the replay harness work unchanged.
    def read(self, address, size):
        value = 0
        for i in range(size):
            index = self._index_for(address + i)
            if index is not None:
                value |= (self.read_index(index) & 0xFF) << (8 * i)
        self._record('r', address, size, value)
        return value.to_bytes(size, 'little')

    def write(self, address, size, value):
        if isinstance(value, (bytes, bytearray)):
            value = int.from_bytes(value, 'little')
        self._record('w', address, size, value)
        for i in range(size):
            index = self._index_for(address + i)
            if index is not None:
                self.write_index(index, (value >> (8 * i)) & 0xFF)

    # MMIODevice declares these abstract; route them through the index
    # decode so direct callers and UnimplementedDevice-style use still work.
    def read_register(self, offset):
        index = self._index_for(offset)
        return self.DEFAULT_READ if index is None else self.read_index(index)

    def write_register(self, offset, value):
        index = self._index_for(offset)
        if index is not None:
            self.write_index(index, value)


class TegraUart(Uart8250):
    """
    Tegra X1 (T210) UART, as found on the Jetson Nano's debug console.

    Fixes the 4-byte register spacing Tegra uses. The Nano's console is
    ``UART-A`` at 0x70006000; UART-B, C and D follow at 0x70006040,
    0x70006200 and 0x70006300.

    Tegra adds registers above the 16550 set (``IRDA_CSR`` and friends) that
    are not modelled; they read as zero.
    """

    #: Input clock to UART-A: a 408 MHz PLL, as the vendor tables have it.
    CLOCK = 408000000

    def __init__(self, name='uarta', trace=False):
        super().__init__(name=name, shift=2, trace=trace)
