import pytest

from armulator.boards import JetsonNano, RaspberryPi3
from armulator.boards.interconnect import SpiBridge
from armulator.peripherals.serial_bus import SpiLoopback, SpiSlaveDevice
from armulator.peripherals.spi_slave import (
    CR_EN, CR_RXE, CR_SPI, CR_TXE, SLV_CR, SLV_SLV, address_octet,
)
from armulator.peripherals.spi_tegra import (
    CMD1_CS_SEL_SHIFT, CMD1_M_S, CMD1_MODE_SHIFT, CMD1_PIO, CMD1_RX_EN,
    CMD1_TX_EN, FIFO_ERR, FIFO_RX_EMPTY, FIFO_RX_UNF, FIFO_RX_FLUSH,
    FIFO_TX_EMPTY, FIFO_TX_FLUSH, FIFO_TX_UNF, SPI_COMMAND1, SPI_DMA_BLK,
    SPI_FIFO_STATUS, SPI_RX_FIFO, SPI_TRANS_STATUS, SPI_TX_FIFO, TRANS_RDY,
    Tegra210Spi,
)

MASTER8 = CMD1_M_S | CMD1_TX_EN | CMD1_RX_EN | 7      # master, 8-bit words


@pytest.fixture
def spi():
    """A Tegra SPI master configured for 8-bit words, TX and RX enabled."""
    controller = Tegra210Spi()
    controller.write_register(SPI_COMMAND1, MASTER8)
    return controller


def transfer(spi, payload):
    """Push ``payload``, set the block count, and trigger the transfer."""
    for byte in payload:
        spi.write_register(SPI_TX_FIFO, byte)
    spi.write_register(SPI_DMA_BLK, len(payload) - 1)
    spi.write_register(SPI_COMMAND1, spi.read_register(SPI_COMMAND1) | CMD1_PIO)


class TestConfiguration:

    def test_bit_length_is_stored_minus_one(self, spi):
        assert spi.bit_length == 8                    # register holds 7
        spi.write_register(SPI_COMMAND1, CMD1_M_S | 15)
        assert spi.bit_length == 16

    def test_master_slave_bit(self, spi):
        assert spi.is_master is True
        spi.write_register(SPI_COMMAND1, MASTER8 & ~CMD1_M_S)
        assert spi.is_master is False

    def test_direction_enables(self, spi):
        assert spi.tx_enabled and spi.rx_enabled
        spi.write_register(SPI_COMMAND1, CMD1_M_S | 7)
        assert not spi.tx_enabled and not spi.rx_enabled

    def test_chip_select_field(self, spi):
        spi.write_register(SPI_COMMAND1, MASTER8 | (2 << CMD1_CS_SEL_SHIFT))
        assert spi.chip_select == 2

    def test_mode_field(self, spi):
        spi.write_register(SPI_COMMAND1, MASTER8 | (3 << CMD1_MODE_SHIFT))
        assert spi.mode == 3

    def test_block_count_is_stored_minus_one(self, spi):
        spi.write_register(SPI_DMA_BLK, 3)
        assert spi.block_count == 4

    def test_pio_is_a_strobe_not_stored(self, spi):
        spi.write_register(SPI_COMMAND1, MASTER8 | CMD1_PIO)
        assert not spi.read_register(SPI_COMMAND1) & CMD1_PIO


class TestTransfer:

    def test_nothing_moves_until_pio_is_set(self, spi):
        # The key difference from the Broadcom controller: filling the TX
        # FIFO does not transmit.
        slave = spi.attach_slave(SpiSlaveDevice())
        spi.write_register(SPI_TX_FIFO, 0x42)
        spi.write_register(SPI_DMA_BLK, 0)
        assert slave.received == b''
        spi.write_register(SPI_COMMAND1, MASTER8 | CMD1_PIO)
        assert slave.received == b'\x42'

    def test_block_count_bounds_the_transfer(self, spi):
        slave = spi.attach_slave(SpiSlaveDevice())
        for byte in (0x11, 0x22, 0x33):
            spi.write_register(SPI_TX_FIFO, byte)
        spi.write_register(SPI_DMA_BLK, 1)            # two packets only
        spi.write_register(SPI_COMMAND1, MASTER8 | CMD1_PIO)
        assert slave.received == b'\x11\x22'
        assert spi.pending_transmit == b'\x33'        # left in the FIFO

    def test_received_data_lands_in_rx_fifo(self, spi):
        spi.attach_slave(SpiSlaveDevice(responses=b'\xAA\xBB'))
        transfer(spi, [0x00, 0x00])
        assert spi.read_register(SPI_RX_FIFO) == 0xAA
        assert spi.read_register(SPI_RX_FIFO) == 0xBB

    def test_loopback(self, spi):
        spi.attach_slave(SpiLoopback())
        transfer(spi, [0x01, 0x02, 0x03])
        assert [spi.read_register(SPI_RX_FIFO) for _ in range(3)] == [1, 2, 3]

    def test_rx_disabled_discards_incoming(self, spi):
        spi.attach_slave(SpiLoopback())
        spi.write_register(SPI_COMMAND1, CMD1_M_S | CMD1_TX_EN | 7)
        transfer(spi, [0x55])
        assert spi.received == b''

    def test_slave_mode_does_not_drive_the_bus(self, spi):
        slave = spi.attach_slave(SpiSlaveDevice())
        spi.write_register(SPI_COMMAND1, (MASTER8 & ~CMD1_M_S))
        transfer(spi, [0x42])
        assert slave.received == b''

    def test_chip_select_routes_to_the_right_slave(self, spi):
        a = spi.attach_slave(SpiSlaveDevice(name='a'), 0)
        b = spi.attach_slave(SpiSlaveDevice(name='b'), 1)
        transfer(spi, [0x11])
        spi.write_register(SPI_COMMAND1, MASTER8 | (1 << CMD1_CS_SEL_SHIFT))
        transfer(spi, [0x22])
        assert a.received == b'\x11'
        assert b.received == b'\x22'

    def test_no_slave_reads_back_zero(self, spi):
        transfer(spi, [0x55])
        assert spi.read_register(SPI_RX_FIFO) == 0


class TestStatus:

    def test_ready_is_set_after_transfer(self, spi):
        spi.attach_slave(SpiLoopback())
        assert not spi.read_register(SPI_TRANS_STATUS) & TRANS_RDY
        transfer(spi, [0x01])
        assert spi.read_register(SPI_TRANS_STATUS) & TRANS_RDY

    def test_ready_is_write_one_to_clear(self, spi):
        spi.attach_slave(SpiLoopback())
        transfer(spi, [0x01])
        spi.write_register(SPI_TRANS_STATUS, TRANS_RDY)
        assert not spi.read_register(SPI_TRANS_STATUS) & TRANS_RDY

    def test_fifo_empty_flags(self, spi):
        status = spi.read_register(SPI_FIFO_STATUS)
        assert status & FIFO_TX_EMPTY and status & FIFO_RX_EMPTY
        spi.write_register(SPI_TX_FIFO, 0x01)
        assert not spi.read_register(SPI_FIFO_STATUS) & FIFO_TX_EMPTY

    def test_fifo_counts_are_reported(self, spi):
        from armulator.peripherals.spi_tegra import (
            FIFO_RX_FULL_COUNT_SHIFT, FIFO_TX_EMPTY_COUNT_SHIFT,
        )
        spi.attach_slave(SpiLoopback())
        transfer(spi, [0x01, 0x02])
        status = spi.read_register(SPI_FIFO_STATUS)
        assert (status >> FIFO_RX_FULL_COUNT_SHIFT) & 0x7F == 2
        empty = (status >> FIFO_TX_EMPTY_COUNT_SHIFT) & 0x7F
        assert empty == spi.fifo_depth

    def test_tx_underrun_when_fifo_is_short(self, spi):
        spi.attach_slave(SpiLoopback())
        spi.write_register(SPI_DMA_BLK, 2)            # ask for three
        spi.write_register(SPI_TX_FIFO, 0x01)         # supply one
        spi.write_register(SPI_COMMAND1, MASTER8 | CMD1_PIO)
        status = spi.read_register(SPI_FIFO_STATUS)
        assert status & FIFO_TX_UNF and status & FIFO_ERR

    def test_rx_underrun_on_empty_read(self, spi):
        spi.read_register(SPI_RX_FIFO)
        assert spi.read_register(SPI_FIFO_STATUS) & FIFO_RX_UNF

    def test_errors_are_write_one_to_clear(self, spi):
        spi.read_register(SPI_RX_FIFO)                # provoke an underrun
        spi.write_register(SPI_FIFO_STATUS, FIFO_RX_UNF | FIFO_ERR)
        assert not spi.read_register(SPI_FIFO_STATUS) & FIFO_RX_UNF

    def test_fifo_flush_bits(self, spi):
        spi.attach_slave(SpiLoopback())
        transfer(spi, [0x01])
        spi.write_register(SPI_TX_FIFO, 0x02)
        spi.write_register(SPI_FIFO_STATUS, FIFO_TX_FLUSH | FIFO_RX_FLUSH)
        assert spi.pending_transmit == b'' and spi.received == b''


class TestBoardIntegration:

    def test_jetson_uses_the_real_tegra_controller(self):
        board = JetsonNano()
        assert isinstance(board.spi, Tegra210Spi)

    def test_mapped_at_spi1_base(self):
        board = JetsonNano()
        bases = {mc.mem: mc.beginning for mc in board.cpu.mem.memories}
        assert bases[board.spi] == 0x7000D400

    def test_broadcom_firmware_does_not_work_on_tegra(self):
        # A driver written for the BCM2835 controller writes CS then FIFO
        # and expects a byte to move. On Tegra nothing happens until PIO.
        board = JetsonNano()
        slave = board.spi.attach_slave(SpiSlaveDevice())
        board.spi.write_register(0x00, 0x80)          # BCM CS: TA=1
        board.spi.write_register(0x04, 0x42)          # BCM FIFO offset
        assert slave.received == b''


class TestCrossDevice:

    def test_tegra_master_drives_a_pi_slave(self):
        nano, pi = JetsonNano(), RaspberryPi3()
        pi.spi_slave.write_register(SLV_SLV, 0x2A)
        pi.spi_slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE)
        SpiBridge(nano, pi)

        payload = [address_octet(0x2A, read=False), 0xDE, 0xAD]
        nano.spi.write_register(SPI_COMMAND1, MASTER8)
        transfer(nano.spi, payload)
        assert pi.spi_slave.received == b'\xDE\xAD'

    def test_transfer_frames_one_dialogue(self):
        nano, pi = JetsonNano(), RaspberryPi3()
        pi.spi_slave.write_register(SLV_SLV, 0x2A)
        pi.spi_slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE)
        SpiBridge(nano, pi)
        nano.spi.write_register(SPI_COMMAND1, MASTER8)
        transfer(nano.spi, [address_octet(0x2A, read=False), 0x01])
        transfer(nano.spi, [address_octet(0x2A, read=False), 0x02])
        assert len(pi.spi_slave.dialogues) == 2

    def test_read_dialogue_returns_data_to_tegra(self):
        nano, pi = JetsonNano(), RaspberryPi3()
        pi.spi_slave.write_register(SLV_SLV, 0x2A)
        pi.spi_slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE)
        bridge = SpiBridge(nano, pi)
        bridge.queue_slave_response(b'\x5A')

        nano.spi.write_register(SPI_COMMAND1, MASTER8)
        transfer(nano.spi, [address_octet(0x2A, read=True), 0x00])
        nano.spi.read_register(SPI_RX_FIFO)           # address octet echo
        assert nano.spi.read_register(SPI_RX_FIFO) == 0x5A
