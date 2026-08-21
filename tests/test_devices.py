import pytest

from armulator.boards import JetsonNano, RaspberryPi4
from armulator.boards.firmware import HAVE_KEYSTONE, firmware
from armulator.peripherals.gpio_tegra import TegraGpio
from armulator.peripherals.uart_pl011 import (
    FR_RXFE, FR_TXFE, INT_RX, BcmSystemTimer, Pl011Uart,
)

needs_keystone = pytest.mark.skipif(
    not HAVE_KEYSTONE, reason='keystone-engine required to assemble firmware'
)


def run(board, source):
    code = firmware(source, address=board.CODE_BASE)
    board.load(board.CODE_BASE, code)
    board.start()
    board.run(5000)
    return board


class TestPl011Uart:

    def test_transmitted_bytes_are_captured(self):
        uart = Pl011Uart()
        for ch in b'hi':
            uart.write_register(0x00, ch)
        assert uart.tx_buffer == b'hi'
        assert uart.text == 'hi'

    def test_flag_register_reports_empty_tx(self):
        uart = Pl011Uart()
        assert uart.read_register(0x18) & FR_TXFE

    def test_rx_fifo_empties_on_read(self):
        uart = Pl011Uart()
        uart.feed('AB')
        assert not uart.read_register(0x18) & FR_RXFE
        assert uart.read_register(0x00) == ord('A')
        assert uart.read_register(0x00) == ord('B')
        assert uart.read_register(0x18) & FR_RXFE

    def test_rx_interrupt_respects_mask(self):
        uart = Pl011Uart()
        uart.feed('x')
        assert uart.irq_pending is False        # masked by default
        uart.write_register(0x38, INT_RX)       # IMSC
        assert uart.irq_pending is True

    def test_icr_clears_interrupt(self):
        uart = Pl011Uart()
        uart.write_register(0x38, INT_RX)
        uart.feed('x')
        assert uart.irq_pending is True
        uart.read_register(0x00)                # drain the FIFO
        uart.write_register(0x44, INT_RX)       # ICR
        assert uart.irq_pending is False

    def test_baud_divisors_are_readable(self):
        uart = Pl011Uart()
        uart.write_register(0x24, 26)           # IBRD
        uart.write_register(0x28, 3)            # FBRD
        assert uart.baud_divisors == (26, 3)

    def test_tx_callback_fires(self):
        uart = Pl011Uart()
        seen = []
        uart.tx_callbacks.append(seen.append)
        uart.write_register(0x00, 0x41)
        assert seen == [0x41]


class TestSystemTimer:

    def test_counter_advances_on_tick(self):
        timer = BcmSystemTimer(auto_advance=0)
        assert timer.read_register(0x04) == 0
        timer.tick(100)
        assert timer.read_register(0x04) == 100

    def test_auto_advance_prevents_infinite_delay_loops(self):
        timer = BcmSystemTimer(auto_advance=5)
        first = timer.read_register(0x04)
        second = timer.read_register(0x04)
        assert second > first

    def test_compare_match_raises_irq(self):
        timer = BcmSystemTimer(auto_advance=0)
        timer.write_register(0x0C, 50)          # C0
        timer.tick(49)
        assert timer.irq_pending is False
        timer.tick(1)
        assert timer.irq_pending is True

    def test_cs_is_write_one_to_clear(self):
        timer = BcmSystemTimer(auto_advance=0)
        timer.write_register(0x0C, 10)
        timer.tick(10)
        assert timer.read_register(0x00) & 1
        timer.write_register(0x00, 1)
        assert timer.read_register(0x00) == 0
        assert timer.irq_pending is False

    def test_high_word_tracks_rollover(self):
        timer = BcmSystemTimer(auto_advance=0)
        timer.tick(0x1_0000_0000 + 7)
        assert timer.read_register(0x08) == 1
        assert timer.read_register(0x04) == 7


class TestTegraGpio:

    def test_pin_name_parsing(self):
        assert TegraGpio.pin_number('PA0') == 0
        assert TegraGpio.pin_number('PB0') == 8
        assert TegraGpio.pin_number('PZ7') == 207
        assert TegraGpio.pin_number('PAA0') == 208
        assert TegraGpio.pin_number('PBB4') == 220

    def test_bad_pin_names_rejected(self):
        for bad in ('PA8', 'PAB0', 'P0', 'PA'):
            with pytest.raises(ValueError):
                TegraGpio.pin_number(bad)

    def test_pin_needs_gpio_mode_and_output_enable(self):
        gpio = TegraGpio()
        gpio.write_register(0x20, 1 << 0)       # OUT port A, pin 0
        assert gpio.level('PA0') is False       # still SFIO, not driving
        gpio.write_register(0x00, 1 << 0)       # CNF -> GPIO mode
        assert gpio.level('PA0') is False       # still an input
        gpio.write_register(0x10, 1 << 0)       # OE -> output
        assert gpio.level('PA0') is True

    def test_masked_write_updates_single_pin(self):
        gpio = TegraGpio()
        gpio.write_register(0x00, 0xFF)         # all of port A to GPIO
        gpio.write_register(0x10, 0xFF)         # all outputs
        # MSK_OUT: mask pin 3 only, set it high, leave the rest alone.
        gpio.write_register(0xA0, (1 << (8 + 3)) | (1 << 3))
        assert gpio.level('PA3') is True
        assert gpio.level('PA2') is False

    def test_masked_write_ignores_unmasked_bits(self):
        gpio = TegraGpio()
        gpio.write_register(0x00, 0xFF)
        gpio.write_register(0x10, 0xFF)
        gpio.write_register(0x20, 0xFF)         # all high
        # Mask only pin 0, write zeros everywhere.
        gpio.write_register(0xA0, (1 << 8) | 0x00)
        assert gpio.level('PA0') is False
        assert gpio.level('PA1') is True        # untouched

    def test_controller_stride_separates_ports(self):
        gpio = TegraGpio()
        # Controller 1 starts at 0x100 and covers ports E-H (pins 32+).
        gpio.write_register(0x100, 1 << 0)
        gpio.write_register(0x110, 1 << 0)
        gpio.write_register(0x120, 1 << 0)
        assert gpio.level('PE0') is True
        assert gpio.level('PA0') is False

    def test_input_register_reads_external_drive(self):
        gpio = TegraGpio()
        gpio.write_register(0x00, 1 << 2)       # PA2 GPIO mode, input
        gpio.drive_input('PA2', True)
        assert gpio.read_register(0x30) & (1 << 2)   # IN

    def test_interrupt_latches_and_clears(self):
        gpio = TegraGpio()
        gpio.write_register(0x00, 1 << 5)
        gpio.write_register(0x50, 1 << 5)       # INT_ENB
        gpio.drive_input('PA5', True)
        assert gpio.read_register(0x40) & (1 << 5)
        assert gpio.irq_pending is True
        gpio.write_register(0x70, 1 << 5)       # INT_CLR
        assert gpio.irq_pending is False


class TestBoardWiring:

    def test_devices_land_at_documented_addresses(self):
        pi4 = RaspberryPi4()
        bases = {mc.mem: mc.beginning for mc in pi4.cpu.mem.memories}
        assert bases[pi4.gpio] == 0xFE200000
        assert bases[pi4.uart] == 0xFE201000
        assert bases[pi4.timer] == 0xFE003000

    def test_jetson_gpio_base(self):
        nano = JetsonNano()
        bases = {mc.mem: mc.beginning for mc in nano.cpu.mem.memories}
        assert bases[nano.gpio] == 0x6000D000

    def test_attach_rejects_ambiguous_placement(self):
        board = RaspberryPi4()
        with pytest.raises(ValueError):
            board.attach('x', Pl011Uart(), offset=0x1000, address=0x2000)

    def test_load_outside_ram_is_rejected(self):
        board = RaspberryPi4()
        with pytest.raises(ValueError):
            board.load(0xF0000000, b'\x00' * 4)


@needs_keystone
class TestFirmwareIntegration:

    def test_firmware_prints_over_uart(self):
        board = run(RaspberryPi4(), """
            ldr r0, =0xFE201000
            mov r1, #0x4F
            str r1, [r0]
            mov r1, #0x4B
            str r1, [r0]
        """)
        assert board.uart.text == 'OK'

    def test_firmware_polls_flag_register_before_writing(self):
        board = run(RaspberryPi4(trace=True), """
            ldr r0, =0xFE201000
        wait:
            ldr r1, [r0, #0x18]        @ FR
            tst r1, #0x20              @ TXFF
            bne wait
            mov r1, #0x41
            str r1, [r0]
        """)
        assert board.uart.text == 'A'
        assert board.uart.reads_of('FR')

    def test_firmware_delay_loop_terminates(self):
        board = run(RaspberryPi4(), """
            ldr r0, =0xFE003000
            ldr r1, [r0, #0x04]        @ CLO
            add r1, r1, #20            @ deadline
        spin:
            ldr r2, [r0, #0x04]
            cmp r2, r1
            blo spin
        """)
        assert board.timer.counter >= 20

    def test_gpio_interrupt_reaches_cpu(self):
        # The Pi 4 routes device interrupts through its GIC-400, so the
        # distributor and CPU interface must be enabled and the interrupt
        # unmasked before a GPIO edge can reach the core -- exactly as on
        # real BCM2711 silicon.  See test_gic400.py for the full handshake.
        from armulator.peripherals.gic400 import (
            GICC_CTLR, GICC_PMR, GICD_CTLR, GICD_ISENABLER, SPI_BASE,
        )
        board = RaspberryPi4()
        intid = SPI_BASE + RaspberryPi4.GPIO_SPI
        board.gic.write_register(GICD_CTLR, 1)
        board.gic.write_register(GICC_CTLR, 1)
        board.gic.write_register(GICC_PMR, 0xFF)
        board.gic.write_register(
            GICD_ISENABLER + 4 * (intid // 32), 1 << (intid % 32)
        )
        board.gpio.write_register(0x4C, 1 << 7)     # GPREN0 pin 7
        board.cpu.registers.cpsr.i = 0              # unmask IRQs
        board.start()
        board.gpio.drive_input(7, True)
        assert board.pending_irq() == ['gpio']
        assert board.service_interrupts() is True
        assert board.cpu.registers.cpsr.m == 0b10010   # IRQ mode

    def test_gpio_interrupt_blocked_by_disabled_gic(self):
        # The same edge, with the GIC left at its reset state, must not
        # reach the CPU.
        board = RaspberryPi4()
        board.gpio.write_register(0x4C, 1 << 7)
        board.cpu.registers.cpsr.i = 0
        board.start()
        board.gpio.drive_input(7, True)
        assert board.pending_irq() == ['gpio']       # device is asserting
        assert board.service_interrupts() is False   # but the GIC gates it

    def test_masked_irq_is_not_delivered(self):
        board = RaspberryPi4()
        board.gpio.write_register(0x4C, 1 << 7)
        board.cpu.registers.cpsr.i = 1              # masked
        board.gpio.drive_input(7, True)
        assert board.pending_irq() == ['gpio']
        assert board.service_interrupts() is False

    def test_jetson_firmware_drives_pin(self):
        board = run(JetsonNano(), """
            ldr r0, =0x6000D000
            mov r1, #1
            str r1, [r0, #0x00]        @ CNF port A -> GPIO
            str r1, [r0, #0x10]        @ OE  -> output
            str r1, [r0, #0x20]        @ OUT -> high
        """)
        assert board.gpio.level('PA0') is True
