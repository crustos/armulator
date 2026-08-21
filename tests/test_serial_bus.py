import pytest

from armulator.boards import RaspberryPi4
from armulator.boards.firmware import HAVE_KEYSTONE, firmware
from armulator.peripherals.serial_bus import (
    BSC_A, BSC_C, BSC_DLEN, BSC_FIFO, BSC_S, C_I2CEN, C_INTD, C_READ, C_ST,
    CS_CLEAR_RX, CS_DONE, CS_INTR, CS_RXD, CS_TA, S_DONE, S_ERR, S_RXD,
    SPI_CLK, SPI_CS, SPI_DLEN, SPI_FIFO, Bcm2835I2c, Bcm2835Spi,
    I2cSlaveDevice, SpiLoopback, SpiSlaveDevice,
)

needs_keystone = pytest.mark.skipif(
    not HAVE_KEYSTONE, reason='keystone-engine required to assemble firmware'
)

PI_SPI = 0xFE204000
PI_I2C = 0xFE804000


def run(board, source, budget=5000):
    board.load(board.CODE_BASE, firmware(source, address=board.CODE_BASE))
    board.start()
    board.run(budget)
    return board


class TestSpi:

    def test_transfer_is_full_duplex(self):
        spi = Bcm2835Spi()
        spi.attach_slave(SpiSlaveDevice(responses=b'\xAA'))
        spi.write_register(SPI_FIFO, 0x55)
        assert spi.transmitted == b'\x55'
        assert spi.read_register(SPI_FIFO) == 0xAA

    def test_loopback_returns_what_was_sent(self):
        spi = Bcm2835Spi()
        spi.attach_slave(SpiLoopback())
        for byte in (0x01, 0x02, 0x03):
            spi.write_register(SPI_FIFO, byte)
        assert [spi.read_register(SPI_FIFO) for _ in range(3)] == [1, 2, 3]

    def test_slave_records_what_master_sent(self):
        spi = Bcm2835Spi()
        slave = spi.attach_slave(SpiSlaveDevice())
        for byte in b'hi':
            spi.write_register(SPI_FIFO, byte)
        assert slave.received == b'hi'

    def test_no_slave_reads_as_zero(self):
        spi = Bcm2835Spi()
        spi.write_register(SPI_FIFO, 0x55)
        assert spi.read_register(SPI_FIFO) == 0x00

    def test_chip_select_routes_to_the_right_slave(self):
        spi = Bcm2835Spi()
        a = spi.attach_slave(SpiSlaveDevice(name='a'), chip_select=0)
        b = spi.attach_slave(SpiSlaveDevice(name='b'), chip_select=1)
        spi.write_register(SPI_CS, 0)
        spi.write_register(SPI_FIFO, 0x11)
        spi.write_register(SPI_CS, 1)
        spi.write_register(SPI_FIFO, 0x22)
        assert a.received == b'\x11'
        assert b.received == b'\x22'

    def test_rxd_flag_tracks_fifo(self):
        spi = Bcm2835Spi()
        spi.attach_slave(SpiLoopback())
        assert not spi.read_register(SPI_CS) & CS_RXD
        spi.write_register(SPI_FIFO, 0x01)
        assert spi.read_register(SPI_CS) & CS_RXD
        spi.read_register(SPI_FIFO)
        assert not spi.read_register(SPI_CS) & CS_RXD

    def test_done_reflects_transfer_active(self):
        spi = Bcm2835Spi()
        spi.write_register(SPI_CS, CS_TA)
        assert not spi.read_register(SPI_CS) & CS_DONE
        spi.write_register(SPI_CS, 0)
        assert spi.read_register(SPI_CS) & CS_DONE

    def test_clear_rx_is_a_strobe_not_state(self):
        spi = Bcm2835Spi()
        spi.attach_slave(SpiLoopback())
        spi.write_register(SPI_FIFO, 0x01)
        spi.write_register(SPI_CS, CS_CLEAR_RX)
        assert not spi.read_register(SPI_CS) & CS_RXD
        # The clear bit must not persist in the register.
        assert not spi.read_register(SPI_CS) & CS_CLEAR_RX

    def test_interrupt_on_receive_when_enabled(self):
        spi = Bcm2835Spi()
        spi.attach_slave(SpiLoopback())
        spi.write_register(SPI_CS, CS_TA)
        spi.write_register(SPI_FIFO, 0x01)
        assert spi.irq_pending is False           # INTR not set
        spi.write_register(SPI_CS, CS_TA | CS_INTR)
        assert spi.irq_pending is True

    def test_clock_divider_round_trips(self):
        spi = Bcm2835Spi()
        spi.write_register(SPI_CLK, 250)
        assert spi.clock_divider == 250


class TestI2c:

    def test_write_reaches_slave(self):
        i2c = Bcm2835I2c()
        slave = i2c.attach_slave(I2cSlaveDevice(address=0x48))
        i2c.write_register(BSC_A, 0x48)
        i2c.write_register(BSC_FIFO, 0x10)        # register pointer
        i2c.write_register(BSC_FIFO, 0xAB)        # value
        i2c.write_register(BSC_DLEN, 2)
        i2c.write_register(BSC_C, C_I2CEN | C_ST)
        assert slave.registers[0x10] == 0xAB

    def test_read_returns_slave_registers(self):
        i2c = Bcm2835I2c()
        i2c.attach_slave(I2cSlaveDevice(address=0x48, registers={0: 0xDE, 1: 0xAD}))
        i2c.write_register(BSC_A, 0x48)
        i2c.write_register(BSC_DLEN, 2)
        i2c.write_register(BSC_C, C_I2CEN | C_READ | C_ST)
        assert i2c.read_register(BSC_FIFO) == 0xDE
        assert i2c.read_register(BSC_FIFO) == 0xAD

    def test_missing_slave_sets_error(self):
        i2c = Bcm2835I2c()
        i2c.write_register(BSC_A, 0x77)           # nobody at this address
        i2c.write_register(BSC_DLEN, 1)
        i2c.write_register(BSC_C, C_I2CEN | C_ST)
        assert i2c.read_register(BSC_S) & S_ERR

    def test_error_is_write_one_to_clear(self):
        i2c = Bcm2835I2c()
        i2c.write_register(BSC_A, 0x77)
        i2c.write_register(BSC_DLEN, 1)
        i2c.write_register(BSC_C, C_I2CEN | C_ST)
        i2c.write_register(BSC_S, S_ERR)
        assert not i2c.read_register(BSC_S) & S_ERR

    def test_done_is_set_after_transfer(self):
        i2c = Bcm2835I2c()
        i2c.attach_slave(I2cSlaveDevice(address=0x48))
        i2c.write_register(BSC_A, 0x48)
        i2c.write_register(BSC_DLEN, 0)
        i2c.write_register(BSC_C, C_I2CEN | C_ST)
        assert i2c.read_register(BSC_S) & S_DONE

    def test_st_is_a_strobe(self):
        i2c = Bcm2835I2c()
        i2c.attach_slave(I2cSlaveDevice(address=0x48))
        i2c.write_register(BSC_A, 0x48)
        i2c.write_register(BSC_C, C_I2CEN | C_ST)
        assert not i2c.read_register(BSC_C) & C_ST

    def test_dlen_bounds_the_write(self):
        i2c = Bcm2835I2c()
        slave = i2c.attach_slave(I2cSlaveDevice(address=0x48))
        i2c.write_register(BSC_A, 0x48)
        for byte in (0x00, 0x11, 0x22, 0x33):
            i2c.write_register(BSC_FIFO, byte)
        i2c.write_register(BSC_DLEN, 2)           # only two bytes go out
        i2c.write_register(BSC_C, C_I2CEN | C_ST)
        assert slave.received == [b'\x00\x11']

    def test_interrupt_on_done_when_enabled(self):
        i2c = Bcm2835I2c()
        i2c.attach_slave(I2cSlaveDevice(address=0x48))
        i2c.write_register(BSC_A, 0x48)
        i2c.write_register(BSC_DLEN, 0)
        i2c.write_register(BSC_C, C_I2CEN | C_INTD | C_ST)
        assert i2c.irq_pending is True
        i2c.write_register(BSC_S, S_DONE)
        assert i2c.irq_pending is False

    def test_rxd_flag_tracks_receive_fifo(self):
        i2c = Bcm2835I2c()
        i2c.attach_slave(I2cSlaveDevice(address=0x48, registers={0: 0x5A}))
        i2c.write_register(BSC_A, 0x48)
        i2c.write_register(BSC_DLEN, 1)
        i2c.write_register(BSC_C, C_I2CEN | C_READ | C_ST)
        assert i2c.read_register(BSC_S) & S_RXD
        i2c.read_register(BSC_FIFO)
        assert not i2c.read_register(BSC_S) & S_RXD


class TestBoardWiring:

    def test_buses_land_at_documented_addresses(self):
        board = RaspberryPi4()
        bases = {mc.mem: mc.beginning for mc in board.cpu.mem.memories}
        assert bases[board.spi] == 0xFE204000
        assert bases[board.i2c] == 0xFE804000


@needs_keystone
class TestFirmware:

    def test_firmware_shifts_bytes_over_spi(self):
        board = RaspberryPi4(trace=True)
        slave = board.spi.attach_slave(SpiSlaveDevice(responses=b'\x99'))
        run(board, f"""
            ldr r0, ={PI_SPI:#x}
            mov r1, #0x80              @ TA=1
            str r1, [r0, #0x00]
            mov r2, #0x42
            str r2, [r0, #0x04]
            ldr r3, [r0, #0x04]        @ read the byte shifted back
        """)
        assert slave.received == b'\x42'
        assert board.cpu.registers.get(3) == 0x99

    def test_firmware_polls_rxd_before_reading(self):
        board = RaspberryPi4(trace=True)
        board.spi.attach_slave(SpiLoopback())
        run(board, f"""
            ldr r0, ={PI_SPI:#x}
            mov r1, #0x80
            str r1, [r0, #0x00]
            mov r2, #0x7E
            str r2, [r0, #0x04]
        wait_rxd:
            ldr r3, [r0, #0x00]
            tst r3, #(1 << 17)         @ CS_RXD
            beq wait_rxd
            ldr r4, [r0, #0x04]
        """)
        assert board.cpu.registers.get(4) == 0x7E

    def test_firmware_writes_i2c_register(self):
        board = RaspberryPi4(trace=True)
        slave = board.i2c.attach_slave(I2cSlaveDevice(address=0x48))
        run(board, f"""
            ldr r0, ={PI_I2C:#x}
            mov r1, #0x48
            str r1, [r0, #0x0C]        @ A
            mov r2, #0x01
            str r2, [r0, #0x10]        @ FIFO: register 1
            mov r2, #0xEE
            str r2, [r0, #0x10]        @ FIFO: value
            mov r3, #2
            str r3, [r0, #0x08]        @ DLEN
            mov r4, #0x8080            @ I2CEN | ST
            str r4, [r0, #0x00]        @ C
        """)
        assert slave.registers[0x01] == 0xEE

    def test_firmware_detects_nack(self):
        board = RaspberryPi4()
        run(board, f"""
            ldr r0, ={PI_I2C:#x}
            mov r1, #0x77              @ no slave here
            str r1, [r0, #0x0C]
            mov r3, #1
            str r3, [r0, #0x08]
            mov r4, #0x8080
            str r4, [r0, #0x00]
            ldr r5, [r0, #0x04]        @ S
            and r5, r5, #0x100         @ ERR
        """)
        assert board.cpu.registers.get(5) == 0x100
