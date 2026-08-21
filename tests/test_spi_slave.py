import pytest

from armulator.boards import RaspberryPi3, RaspberryPi4
from armulator.boards.firmware import HAVE_KEYSTONE, firmware
from armulator.boards.interconnect import Machine, SpiBridge
from armulator.peripherals.serial_bus import Bcm2835Spi
from armulator.peripherals.spi_slave import (
    CR_BRK, CR_EN, CR_I2C, CR_RXE, CR_SPI, CR_TESTFIFO, CR_TXE, FR_RXFE,
    FR_TXFE, INT_RX, MISO_IDLE, RSR_OE, SLV_CR, SLV_DR, SLV_FR, SLV_ICR,
    SLV_IMSC, SLV_RIS, SLV_RSR, SLV_SLV, SLV_TDR, Bcm2835SpiSlave,
    address_octet,
)

needs_keystone = pytest.mark.skipif(
    not HAVE_KEYSTONE, reason='keystone-engine required to assemble firmware'
)

ADDRESS = 0x2A


@pytest.fixture
def slave():
    """An enabled SPI slave at address 0x2A with TX and RX enabled."""
    s = Bcm2835SpiSlave()
    s.write_register(SLV_SLV, ADDRESS)
    s.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE)
    return s


def write_dialogue(slave, payload, address=ADDRESS):
    slave.select()
    slave.transfer(address_octet(address, read=False))
    out = [slave.transfer(b) for b in payload]
    slave.deselect()
    return out


def read_dialogue(slave, count, address=ADDRESS):
    slave.select()
    slave.transfer(address_octet(address, read=True))
    out = [slave.transfer(0x00) for _ in range(count)]
    slave.deselect()
    return bytes(out)


class TestAddressOctet:

    def test_encoding(self):
        assert address_octet(0x2A, read=False) == 0x54
        assert address_octet(0x2A, read=True) == 0x55

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            address_octet(0x80, read=False)


class TestDialogueProtocol:

    def test_write_dialogue_fills_rx_fifo(self, slave):
        write_dialogue(slave, b'\xDE\xAD')
        assert slave.received == b'\xDE\xAD'

    def test_write_dialogue_drives_nothing_on_miso(self, slave):
        returned = write_dialogue(slave, b'\x01\x02')
        assert returned == [MISO_IDLE, MISO_IDLE]

    def test_address_octet_itself_returns_idle(self, slave):
        slave.select()
        assert slave.transfer(address_octet(ADDRESS, read=False)) == MISO_IDLE

    def test_read_dialogue_serialises_tx_fifo(self, slave):
        slave.queue_transmit(b'\xBE\xEF')
        assert read_dialogue(slave, 2) == b'\xBE\xEF'

    def test_read_dialogue_discards_mosi(self, slave):
        slave.queue_transmit(b'\x11')
        slave.select()
        slave.transfer(address_octet(ADDRESS, read=True))
        slave.transfer(0xAA)                  # master data, must be ignored
        slave.deselect()
        assert slave.received == b''

    def test_read_dialogue_idles_when_tx_fifo_empty(self, slave):
        assert read_dialogue(slave, 2) == bytes([MISO_IDLE, MISO_IDLE])

    def test_dialogue_is_half_duplex(self, slave):
        # A conventional full-duplex slave would return TX data during a
        # write; this block does not, which is the whole point.
        slave.queue_transmit(b'\x99')
        returned = write_dialogue(slave, b'\x01')
        assert returned == [MISO_IDLE]
        assert slave.pending_transmit == b'\x99'   # untouched

    def test_wrong_address_is_ignored(self, slave):
        write_dialogue(slave, b'\xFF', address=0x55)
        assert slave.received == b''

    def test_wrong_address_does_not_consume_tx(self, slave):
        slave.queue_transmit(b'\x77')
        read_dialogue(slave, 1, address=0x55)
        assert slave.pending_transmit == b'\x77'

    def test_reselect_starts_a_new_dialogue(self, slave):
        write_dialogue(slave, b'\x01')
        write_dialogue(slave, b'\x02')
        assert slave.received == b'\x01\x02'
        assert len(slave.dialogues) == 2

    def test_dialogues_are_recorded_with_direction(self, slave):
        write_dialogue(slave, b'\x01')
        slave.queue_transmit(b'\x02')
        read_dialogue(slave, 1)
        assert slave.dialogues[0][1] is False
        assert slave.dialogues[1][1] is True

    def test_transfer_without_select_is_implicitly_framed(self, slave):
        # Masters that never touch TA still work.
        slave.transfer(address_octet(ADDRESS, read=False))
        slave.transfer(0x42)
        assert slave.received == b'\x42'


class TestEnableGating:

    def test_disabled_block_ignores_traffic(self):
        s = Bcm2835SpiSlave()
        s.write_register(SLV_SLV, ADDRESS)
        write_dialogue(s, b'\x01')
        assert s.received == b''

    def test_i2c_mode_does_not_answer_spi(self):
        s = Bcm2835SpiSlave()
        s.write_register(SLV_SLV, ADDRESS)
        s.write_register(SLV_CR, CR_EN | CR_I2C | CR_RXE)
        write_dialogue(s, b'\x01')
        assert s.received == b''

    def test_rxe_gates_reception(self, slave):
        slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_TXE)   # RXE off
        write_dialogue(slave, b'\x01')
        assert slave.received == b''

    def test_txe_gates_transmission(self, slave):
        slave.queue_transmit(b'\x33')
        slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE)   # TXE off
        assert read_dialogue(slave, 1) == bytes([MISO_IDLE])


class TestRegisters:

    def test_dr_is_both_fifo_ports(self, slave):
        # No separate FIFO register: DR writes go to TX, DR reads from RX.
        slave.write_register(SLV_DR, 0x5A)
        assert slave.pending_transmit == b'\x5A'
        write_dialogue(slave, b'\xA5')
        assert slave.read_register(SLV_DR) == 0xA5

    def test_flag_register_tracks_fifos(self, slave):
        assert slave.read_register(SLV_FR) & FR_TXFE
        assert slave.read_register(SLV_FR) & FR_RXFE
        slave.write_register(SLV_DR, 0x01)
        assert not slave.read_register(SLV_FR) & FR_TXFE
        write_dialogue(slave, b'\x02')
        assert not slave.read_register(SLV_FR) & FR_RXFE

    def test_slv_register_is_seven_bit(self, slave):
        slave.write_register(SLV_SLV, 0xFF)
        assert slave.address == 0x7F

    def test_rx_overrun_sets_status(self, slave):
        write_dialogue(slave, bytes(slave.fifo_depth + 4))
        assert slave.read_register(SLV_RSR) & RSR_OE

    def test_overrun_is_write_one_to_clear(self, slave):
        write_dialogue(slave, bytes(slave.fifo_depth + 1))
        slave.write_register(SLV_RSR, RSR_OE)
        assert not slave.read_register(SLV_RSR) & RSR_OE

    def test_rx_interrupt_respects_mask(self, slave):
        write_dialogue(slave, b'\x01')
        assert slave.irq_pending is False
        slave.write_register(SLV_IMSC, INT_RX)
        assert slave.irq_pending is True

    def test_ris_reports_unmasked_state(self, slave):
        write_dialogue(slave, b'\x01')
        assert slave.read_register(SLV_RIS) & INT_RX


class TestErrata:
    """Behaviours where real silicon departs from the datasheet."""

    def test_brk_does_not_clear_fifos_by_default(self, slave):
        # The datasheet says BRK clears the FIFOs.  Hardware ignores it, so
        # firmware relying on it will transmit stale data.
        slave.queue_transmit(b'\xDE\xAD')
        slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_TXE | CR_BRK)
        assert slave.pending_transmit == b'\xDE\xAD'

    def test_brk_can_be_modelled_as_working(self):
        s = Bcm2835SpiSlave(brk_clears_fifos=True)
        s.write_register(SLV_SLV, ADDRESS)
        s.write_register(SLV_CR, CR_EN | CR_SPI | CR_TXE)
        s.queue_transmit(b'\xDE')
        s.write_register(SLV_CR, CR_EN | CR_SPI | CR_TXE | CR_BRK)
        assert s.pending_transmit == b''

    def test_brk_is_a_strobe_not_stored(self, slave):
        slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_BRK)
        assert not slave.read_register(SLV_CR) & CR_BRK

    def test_tdr_peeks_without_draining(self, slave):
        # Reading TDR was suggested as a FIFO-drain workaround; it only
        # peeks, which is why that workaround does not work.
        slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_TXE | CR_TESTFIFO)
        slave.queue_transmit(b'\x11\x22')
        assert slave.read_register(SLV_TDR) == 0x11
        assert slave.pending_transmit == b'\x11\x22'

    def test_miso_idles_high_not_low(self, slave):
        assert MISO_IDLE == 0xFF
        assert write_dialogue(slave, b'\x01') == [0xFF]


class TestMasterFraming:

    def test_master_ta_drives_select(self):
        master = Bcm2835Spi()
        slave = Bcm2835SpiSlave()
        slave.write_register(SLV_SLV, ADDRESS)
        slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE)
        master.attach_slave(slave)
        master.write_register(0x00, 0x80)                 # TA=1
        master.write_register(0x04, address_octet(ADDRESS, read=False))
        master.write_register(0x04, 0x42)
        master.write_register(0x00, 0x00)                 # TA=0
        assert slave.received == b'\x42'
        assert len(slave.dialogues) == 1

    def test_changing_chip_select_reframes(self):
        master = Bcm2835Spi()
        a = Bcm2835SpiSlave(name='a')
        b = Bcm2835SpiSlave(name='b')
        for s in (a, b):
            s.write_register(SLV_SLV, ADDRESS)
            s.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE)
        master.attach_slave(a, 0)
        master.attach_slave(b, 1)
        master.write_register(0x00, 0x80)                 # TA=1, CS=0
        master.write_register(0x04, address_octet(ADDRESS, read=False))
        master.write_register(0x04, 0x11)
        master.write_register(0x00, 0x81)                 # TA=1, CS=1
        master.write_register(0x04, address_octet(ADDRESS, read=False))
        master.write_register(0x04, 0x22)
        master.write_register(0x00, 0x00)
        assert a.received == b'\x11'
        assert b.received == b'\x22'

    def test_slaves_without_framing_hooks_still_work(self):
        from armulator.peripherals.serial_bus import SpiLoopback
        master = Bcm2835Spi()
        master.attach_slave(SpiLoopback())
        master.write_register(0x00, 0x80)
        master.write_register(0x04, 0x77)
        assert master.read_register(0x04) == 0x77


class TestBridge:

    def test_bridge_uses_the_real_slave_controller(self):
        master, slave_board = RaspberryPi4(), RaspberryPi3()
        bridge = SpiBridge(master, slave_board)
        assert bridge.slave is slave_board.spi_slave

    def test_bridge_rejects_board_without_slave(self):
        master, nano = RaspberryPi4(), __import__(
            'armulator.boards', fromlist=['JetsonNano']
        ).JetsonNano()
        with pytest.raises(ValueError):
            SpiBridge(master, nano)

    def test_payload_crosses_boards(self):
        master, slave_board = RaspberryPi4(), RaspberryPi3()
        slave_board.spi_slave.write_register(SLV_SLV, ADDRESS)
        slave_board.spi_slave.write_register(
            SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE
        )
        bridge = SpiBridge(master, slave_board)
        master.spi.write_register(0x00, 0x80)
        master.spi.write_register(0x04, address_octet(ADDRESS, read=False))
        for byte in b'\xDE\xAD':
            master.spi.write_register(0x04, byte)
        master.spi.write_register(0x00, 0x00)
        assert slave_board.spi_slave.received == b'\xDE\xAD'

    def test_bridge_records_both_directions(self):
        master, slave_board = RaspberryPi4(), RaspberryPi3()
        slave_board.spi_slave.write_register(SLV_SLV, ADDRESS)
        slave_board.spi_slave.write_register(
            SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE
        )
        bridge = SpiBridge(master, slave_board)
        bridge.queue_slave_response(b'\xBE')
        master.spi.write_register(0x00, 0x80)
        master.spi.write_register(0x04, address_octet(ADDRESS, read=True))
        master.spi.write_register(0x04, 0x00)
        master.spi.write_register(0x00, 0x00)
        assert bridge.miso.endswith(b'\xBE')
        assert bridge.mosi[0] == address_octet(ADDRESS, read=True)


@needs_keystone
class TestFirmware:

    def test_slave_firmware_configures_block(self):
        board = RaspberryPi4(trace=True)
        base = 0xFE214000
        board.load(board.CODE_BASE, firmware(f"""
            ldr r0, ={base:#x}
            mov r1, #{ADDRESS}
            str r1, [r0, #0x08]        @ SLV
            mov r2, #0x300
            orr r2, r2, #0x3           @ TXE | RXE | SPI | EN
            str r2, [r0, #0x0C]        @ CR
        """, address=board.CODE_BASE))
        board.start()
        board.run(500)
        assert board.spi_slave.address == ADDRESS
        assert board.spi_slave.enabled and board.spi_slave.spi_mode

    def test_slave_firmware_reads_received_byte(self):
        board = RaspberryPi4()
        board.spi_slave.write_register(SLV_SLV, ADDRESS)
        board.spi_slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE)
        write_dialogue(board.spi_slave, b'\x5A')
        base = 0xFE214000
        board.load(board.CODE_BASE, firmware(f"""
            ldr r0, ={base:#x}
        wait:
            ldr r1, [r0, #0x10]        @ FR
            tst r1, #0x8               @ RXFE
            bne wait
            ldr r2, [r0, #0x00]        @ DR
        """, address=board.CODE_BASE))
        board.start()
        board.run(2000)
        assert board.cpu.registers.get(2) == 0x5A

    def test_two_boards_exchange_under_firmware(self):
        master, slave_board = RaspberryPi4(), RaspberryPi3()
        slave_board.load(slave_board.CODE_BASE, firmware(f"""
            ldr r0, ={0x3F214000:#x}
            mov r1, #{ADDRESS}
            str r1, [r0, #0x08]
            mov r2, #0x300
            orr r2, r2, #0x3
            str r2, [r0, #0x0C]
        """, address=slave_board.CODE_BASE))
        slave_board.start()
        slave_board.run(500)

        master.load(master.CODE_BASE, firmware(f"""
            ldr r0, ={0xFE204000:#x}
            mov r1, #0x80
            str r1, [r0, #0x00]        @ TA=1
            mov r2, #{address_octet(ADDRESS, read=False)}
            str r2, [r0, #0x04]
            mov r2, #0x42
            str r2, [r0, #0x04]
            mov r1, #0
            str r1, [r0, #0x00]        @ TA=0
        """, address=master.CODE_BASE))
        master.start()

        machine = Machine()
        machine.add('master', master)
        machine.add('slave', slave_board)
        SpiBridge(master, slave_board)
        machine.run(5000)
        assert slave_board.spi_slave.received == b'\x42'
