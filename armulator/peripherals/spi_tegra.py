"""
Tegra X1 (T210) SPI controller.

Replaces the Broadcom stand-in previously used on the Jetson board.  The
Tegra SPI block shares nothing with Broadcom's beyond being SPI: different
register map, different framing, programmable word length, and transfers
that are triggered explicitly rather than implied by a FIFO write.

The register layout and bit definitions here are taken from the in-tree
drivers -- U-Boot's ``drivers/spi/tegra114_spi.c`` and Linux's
``drivers/spi/spi-tegra114.c`` -- which cover T114/T124/T210/T186 with the
same block.  That is a stronger source than the datasheet for this part,
since the Tegra X1 TRM is available only to registered developers.

HOW A TRANSFER WORKS
--------------------
Unlike the BCM2835 master, where writing the FIFO shifts a byte immediately,
a Tegra transfer is set up and then started:

    1. program ``COMMAND1``: word length, master mode, TX/RX enables, chip
       select
    2. write ``DMA_BLK`` with the packet count minus one
    3. push outgoing words into ``TX_FIFO`` (0x108)
    4. set ``COMMAND1.PIO`` (bit 31) to start
    5. poll ``TRANS_STATUS.RDY``, then drain ``RX_FIFO`` (0x188)

Nothing moves on the bus until step 4.  Firmware ported from a Broadcom
controller that expects a FIFO write to transmit will fill the TX FIFO and
then wait forever -- which this model reproduces rather than papering over.

WORD LENGTH
-----------
``COMMAND1.BIT_LENGTH`` holds *bits minus one*, so an 8-bit word is 7.  Only
whole-byte lengths are modelled here; sub-byte and >8-bit words are accepted
and truncated to bytes, with the configured length available as
:attr:`Tegra210Spi.bit_length` for tests that care.
"""

from armulator.peripherals.mmio import MMIODevice

# Register offsets (U-Boot struct spi_regs, tegra114_spi.c)
SPI_COMMAND1 = 0x000
SPI_COMMAND2 = 0x004
SPI_CS_TIM1 = 0x008
SPI_CS_TIM2 = 0x00C
SPI_TRANS_STATUS = 0x010
SPI_FIFO_STATUS = 0x014
SPI_TX_DATA = 0x018
SPI_RX_DATA = 0x01C
SPI_DMA_CTL = 0x020
SPI_DMA_BLK = 0x024
SPI_TX_FIFO = 0x108
SPI_RX_FIFO = 0x188
SPI_SPARE_CTRL = 0x18C

# COMMAND1 bits
CMD1_BIT_LENGTH_MASK = 0x1F           # bits 4:0, holds length minus one
CMD1_PACKED = 1 << 5
CMD1_TX_EN = 1 << 11
CMD1_RX_EN = 1 << 12
CMD1_CS_SW_VAL = 1 << 20
CMD1_CS_SW_HW = 1 << 21
CMD1_CS_SEL_SHIFT = 26
CMD1_CS_SEL_MASK = 0x3 << CMD1_CS_SEL_SHIFT
CMD1_MODE_SHIFT = 28
CMD1_MODE_MASK = 0x3 << CMD1_MODE_SHIFT
CMD1_M_S = 1 << 30                    # 1 = master, 0 = slave
CMD1_PIO = 1 << 31                    # write 1 to start a transfer

# TRANS_STATUS bits
TRANS_BLK_CNT_MASK = 0xFFFF
TRANS_RDY = 1 << 30                   # transfer complete; write 1 to clear

# FIFO_STATUS bits
FIFO_RX_EMPTY = 1 << 0
FIFO_RX_FULL = 1 << 1
FIFO_TX_EMPTY = 1 << 2
FIFO_TX_FULL = 1 << 3
FIFO_RX_UNF = 1 << 4
FIFO_RX_OVF = 1 << 5
FIFO_TX_UNF = 1 << 6
FIFO_TX_OVF = 1 << 7
FIFO_ERR = 1 << 8
FIFO_TX_FLUSH = 1 << 14
FIFO_RX_FLUSH = 1 << 15
FIFO_TX_EMPTY_COUNT_SHIFT = 16
FIFO_RX_FULL_COUNT_SHIFT = 23


class Tegra210Spi(MMIODevice):
    """
    Tegra X1 SPI master.

    Accepts the same slave objects as the Broadcom controller -- anything
    implementing ``transfer(byte) -> byte``, optionally with ``select`` and
    ``deselect`` for framing -- so a Tegra master can drive a Pi's SPI slave
    block in cross-device tests.

    :param fifo_depth: 64 words on T210 per the driver's flush handling
    """

    SIZE = 0x1000
    FIFO_DEPTH = 64

    REGISTERS = {
        SPI_COMMAND1: 'COMMAND1', SPI_COMMAND2: 'COMMAND2',
        SPI_CS_TIM1: 'CS_TIM1', SPI_CS_TIM2: 'CS_TIM2',
        SPI_TRANS_STATUS: 'TRANS_STATUS', SPI_FIFO_STATUS: 'FIFO_STATUS',
        SPI_TX_DATA: 'TX_DATA', SPI_RX_DATA: 'RX_DATA',
        SPI_DMA_CTL: 'DMA_CTL', SPI_DMA_BLK: 'DMA_BLK',
        SPI_TX_FIFO: 'TX_FIFO', SPI_RX_FIFO: 'RX_FIFO',
        SPI_SPARE_CTRL: 'SPARE_CTRL',
    }

    def __init__(self, name='tegra_spi', trace=False, fifo_depth=FIFO_DEPTH):
        super().__init__(self.SIZE, name=name, trace=trace)
        self.fifo_depth = fifo_depth
        self._command1 = 0
        self._command2 = 0
        self._dma_blk = 0
        self._dma_ctl = 0
        self._cs_tim1 = 0
        self._cs_tim2 = 0
        self._tx = bytearray()
        self._rx = bytearray()
        self._ready = False
        self._errors = 0
        self._selected = False
        #: Chip select line -> slave object.
        self.slaves = {}
        #: Everything shifted out, across all transfers.
        self.transmitted = bytearray()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach_slave(self, slave, chip_select=0):
        """Wire ``slave`` to chip select line ``chip_select`` (0-3)."""
        self.slaves[chip_select] = slave
        return slave

    # ------------------------------------------------------------------
    # Introspection for tests
    # ------------------------------------------------------------------
    @property
    def is_master(self):
        return bool(self._command1 & CMD1_M_S)

    @property
    def tx_enabled(self):
        return bool(self._command1 & CMD1_TX_EN)

    @property
    def rx_enabled(self):
        return bool(self._command1 & CMD1_RX_EN)

    @property
    def bit_length(self):
        """Configured word length in bits (the register holds length - 1)."""
        return (self._command1 & CMD1_BIT_LENGTH_MASK) + 1

    @property
    def chip_select(self):
        return (self._command1 & CMD1_CS_SEL_MASK) >> CMD1_CS_SEL_SHIFT

    @property
    def mode(self):
        """SPI mode 0-3 from COMMAND1.MODE_SEL."""
        return (self._command1 & CMD1_MODE_MASK) >> CMD1_MODE_SHIFT

    @property
    def block_count(self):
        """Packets the next transfer will move (DMA_BLK holds count - 1)."""
        return (self._dma_blk & TRANS_BLK_CNT_MASK) + 1

    @property
    def pending_transmit(self):
        return bytes(self._tx)

    @property
    def received(self):
        return bytes(self._rx)

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------
    def _notify(self, method):
        slave = self.slaves.get(self.chip_select)
        hook = getattr(slave, method, None)
        if hook is not None:
            hook()

    def _start_transfer(self):
        """
        Run the transfer configured in COMMAND1 and DMA_BLK.

        Chip select is asserted for the duration, so slaves that frame their
        protocol on CS (the BCM2835 slave block does) see one dialogue per
        PIO trigger rather than per byte.
        """
        if not self.is_master:
            # Slave mode is not modelled; the transfer simply never completes,
            # which is closer to reality than silently acting as a master.
            return

        slave = self.slaves.get(self.chip_select)
        if not self._selected:
            self._notify('select')
            self._selected = True

        for _ in range(self.block_count):
            if self._tx:
                outgoing = self._tx.pop(0)
            elif self.tx_enabled:
                self._errors |= FIFO_TX_UNF     # underrun: nothing to send
                outgoing = 0x00
            else:
                outgoing = 0x00
            self.transmitted.append(outgoing)

            incoming = slave.transfer(outgoing) if slave is not None else 0x00
            if self.rx_enabled:
                if len(self._rx) < self.fifo_depth:
                    self._rx.append(incoming & 0xFF)
                else:
                    self._errors |= FIFO_RX_OVF

        # CS is released when software clears CS_SW_VAL, but firmware that
        # leaves it alone still gets a clean dialogue boundary per transfer.
        if not (self._command1 & CMD1_CS_SW_HW):
            self._notify('deselect')
            self._selected = False

        self._ready = True

    # ------------------------------------------------------------------
    # Register interface
    # ------------------------------------------------------------------
    def read_register(self, offset):
        if offset == SPI_COMMAND1:
            return self._command1 & ~CMD1_PIO     # PIO is a start strobe
        if offset == SPI_COMMAND2:
            return self._command2
        if offset == SPI_CS_TIM1:
            return self._cs_tim1
        if offset == SPI_CS_TIM2:
            return self._cs_tim2
        if offset == SPI_TRANS_STATUS:
            status = self._dma_blk & TRANS_BLK_CNT_MASK
            if self._ready:
                status |= TRANS_RDY
            return status
        if offset == SPI_FIFO_STATUS:
            status = self._errors
            if not self._rx:
                status |= FIFO_RX_EMPTY
            if len(self._rx) >= self.fifo_depth:
                status |= FIFO_RX_FULL
            if not self._tx:
                status |= FIFO_TX_EMPTY
            if len(self._tx) >= self.fifo_depth:
                status |= FIFO_TX_FULL
            if self._errors:
                status |= FIFO_ERR
            empty_slots = self.fifo_depth - len(self._tx)
            status |= (empty_slots & 0x7F) << FIFO_TX_EMPTY_COUNT_SHIFT
            status |= (len(self._rx) & 0x7F) << FIFO_RX_FULL_COUNT_SHIFT
            return status
        if offset in (SPI_RX_DATA, SPI_RX_FIFO):
            if not self._rx:
                self._errors |= FIFO_RX_UNF
                return 0
            return self._rx.pop(0)
        if offset == SPI_DMA_CTL:
            return self._dma_ctl
        if offset == SPI_DMA_BLK:
            return self._dma_blk
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        if offset == SPI_COMMAND1:
            start = bool(value & CMD1_PIO)
            self._command1 = value & ~CMD1_PIO
            if start:
                self._start_transfer()
        elif offset == SPI_COMMAND2:
            self._command2 = value
        elif offset == SPI_CS_TIM1:
            self._cs_tim1 = value
        elif offset == SPI_CS_TIM2:
            self._cs_tim2 = value
        elif offset == SPI_TRANS_STATUS:
            if value & TRANS_RDY:                 # write 1 to clear
                self._ready = False
        elif offset == SPI_FIFO_STATUS:
            if value & FIFO_TX_FLUSH:
                self._tx.clear()
            if value & FIFO_RX_FLUSH:
                self._rx.clear()
            # Error bits are write-1-to-clear.
            self._errors &= ~(value & (FIFO_RX_UNF | FIFO_RX_OVF |
                                       FIFO_TX_UNF | FIFO_TX_OVF | FIFO_ERR))
        elif offset in (SPI_TX_DATA, SPI_TX_FIFO):
            if len(self._tx) < self.fifo_depth:
                self._tx.append(value & 0xFF)
            else:
                self._errors |= FIFO_TX_OVF
        elif offset == SPI_DMA_CTL:
            self._dma_ctl = value
        elif offset == SPI_DMA_BLK:
            self._dma_blk = value & TRANS_BLK_CNT_MASK
