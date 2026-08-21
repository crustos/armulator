import pytest

from armulator.boards import RaspberryPi3, RaspberryPi4
from armulator.boards.firmware import HAVE_KEYSTONE, firmware
from armulator.peripherals.gpio_bcm import BcmGpio, GpioFunction, Pull

needs_keystone = pytest.mark.skipif(
    not HAVE_KEYSTONE, reason='keystone-engine required to assemble firmware'
)


def run(board, source):
    code = firmware(source, address=board.CODE_BASE)
    board.load(board.CODE_BASE, code)
    board.start()
    board.run(2000)
    return board


# ----------------------------------------------------------------------
# Register-level tests (no CPU involved)
# ----------------------------------------------------------------------
class TestRegisters:

    def test_fsel_selects_function(self):
        gpio = BcmGpio()
        gpio.write_register(0x04, 0b001 << 21)          # GPFSEL1, pin 17
        assert gpio.function(17) == GpioFunction.OUTPUT
        assert gpio.function(16) == GpioFunction.INPUT

    def test_alt_function_encoding(self):
        gpio = BcmGpio()
        # Pin 14 is ALT0 (TXD0) in field 4 of GPFSEL1.
        gpio.write_register(0x04, 0b100 << 12)
        assert gpio.function(14) == GpioFunction.ALT0

    def test_set_and_clear_are_independent_registers(self):
        gpio = BcmGpio()
        gpio.write_register(0x04, 0b001 << 21)
        gpio.write_register(0x1C, 1 << 17)              # GPSET0
        assert gpio.level(17) is True
        gpio.write_register(0x28, 1 << 17)              # GPCLR0
        assert gpio.level(17) is False

    def test_set_register_reads_as_zero(self):
        # GPSET/GPCLR are write-only on real silicon.
        gpio = BcmGpio()
        gpio.write_register(0x1C, 0xFFFFFFFF)
        assert gpio.read_register(0x1C) == 0

    def test_gplev_reflects_driven_inputs(self):
        gpio = BcmGpio()
        gpio.drive_input(3, True)
        gpio.drive_input(4, False)
        level = gpio.read_register(0x34)               # GPLEV0
        assert level & (1 << 3)
        assert not level & (1 << 4)

    def test_pins_above_31_use_bank_one(self):
        gpio = BcmGpio()
        gpio.write_register(0x10, 0b001 << 6)          # GPFSEL4, pin 42
        gpio.write_register(0x20, 1 << (42 - 32))      # GPSET1
        assert gpio.level(42) is True
        assert gpio.read_register(0x38) & (1 << 10)    # GPLEV1

    def test_byte_write_is_read_modify_write(self):
        # Sub-word MMIO access must not clobber the rest of the word.
        # GPFSEL holds 10 pins x 3 bits, so bits 31:30 are reserved and
        # read back as zero -- hence 0x3FFFFF00 rather than 0xFFFFFF00.
        gpio = BcmGpio()
        gpio.write_register(0x04, 0xFFFFFFFF)
        gpio.write(0x04, 1, b'\x00')
        assert gpio.read_register(0x04) == 0x3FFFFF00


class TestPullResistors:

    def test_legacy_pull_needs_clock_sequence(self):
        gpio = BcmGpio(pull_style='legacy')
        gpio.write_register(0x94, int(Pull.UP))        # GPPUD staged only
        assert gpio.pull(5) == Pull.OFF
        gpio.write_register(0x98, 1 << 5)              # GPPUDCLK0 commits it
        assert gpio.pull(5) == Pull.UP

    def test_legacy_pull_only_affects_clocked_pins(self):
        gpio = BcmGpio(pull_style='legacy')
        gpio.write_register(0x94, int(Pull.DOWN))
        gpio.write_register(0x98, 1 << 5)
        assert gpio.pull(5) == Pull.DOWN
        assert gpio.pull(6) == Pull.OFF

    def test_bcm2711_pull_is_direct(self):
        gpio = BcmGpio(pull_style='bcm2711')
        # BCM2711 encoding: 01 = pull-up. Pin 2 -> bits 5:4 of REG0.
        gpio.write_register(0xE4, 0b01 << 4)
        assert gpio.pull(2) == Pull.UP

    def test_bcm2711_encoding_differs_from_legacy(self):
        # 0b10 means pull-up in the legacy register but pull-DOWN on BCM2711.
        gpio = BcmGpio(pull_style='bcm2711')
        gpio.write_register(0xE4, 0b10 << 4)
        assert gpio.pull(2) == Pull.DOWN

    def test_floating_input_follows_pull(self):
        gpio = BcmGpio(pull_style='bcm2711')
        gpio.write_register(0xE4, 0b01 << 4)           # pin 2 pull-up
        assert gpio.level(2) is True                   # floats high
        gpio.drive_input(2, False)
        assert gpio.level(2) is False                  # external driver wins
        gpio.drive_input(2, None)
        assert gpio.level(2) is True                   # released, floats again

    def test_pull_style_is_validated(self):
        with pytest.raises(ValueError):
            BcmGpio(pull_style='bcm9999')


class TestEventDetection:

    def test_rising_edge_latches_and_raises_irq(self):
        gpio = BcmGpio()
        gpio.write_register(0x4C, 1 << 7)              # GPREN0 pin 7
        assert gpio.irq_pending is False
        gpio.drive_input(7, True)
        assert gpio.read_register(0x40) & (1 << 7)     # GPEDS0 latched
        assert gpio.irq_pending is True

    def test_falling_edge_ignored_when_only_rising_enabled(self):
        gpio = BcmGpio()
        gpio.write_register(0x4C, 1 << 7)
        gpio.drive_input(7, True)
        gpio.write_register(0x40, 1 << 7)              # clear
        gpio.drive_input(7, False)
        assert not gpio.read_register(0x40) & (1 << 7)

    def test_eds_is_write_one_to_clear(self):
        gpio = BcmGpio()
        gpio.write_register(0x58, 1 << 9)              # GPFEN0
        gpio.drive_input(9, True)
        gpio.drive_input(9, False)
        assert gpio.read_register(0x40) & (1 << 9)
        gpio.write_register(0x40, 1 << 9)
        assert gpio.read_register(0x40) == 0
        assert gpio.irq_pending is False

    def test_level_detect_is_not_edge_triggered(self):
        gpio = BcmGpio()
        gpio.write_register(0x64, 1 << 11)             # GPHEN0
        gpio.drive_input(11, True)
        gpio.write_register(0x40, 1 << 11)             # clear while still high
        gpio._refresh_events()
        assert gpio.read_register(0x40) & (1 << 11)    # re-asserts


# ----------------------------------------------------------------------
# Firmware-driven tests (real ARM instructions)
# ----------------------------------------------------------------------
@needs_keystone
class TestFirmware:

    def test_pi4_output_high(self):
        board = run(RaspberryPi4(trace=True), """
            ldr r0, =0xFE200000
            mov r1, #1
            lsl r1, r1, #21
            str r1, [r0, #0x04]
            mov r2, #1
            lsl r2, r2, #17
            str r2, [r0, #0x1C]
        """)
        assert board.gpio.function(17) == GpioFunction.OUTPUT
        assert board.gpio.level(17) is True

    def test_pi3_uses_different_peripheral_base(self):
        board = run(RaspberryPi3(trace=True), """
            ldr r0, =0x3F200000
            mov r1, #1
            lsl r1, r1, #21
            str r1, [r0, #0x04]
            mov r2, #1
            lsl r2, r2, #17
            str r2, [r0, #0x1C]
        """)
        assert board.gpio.level(17) is True

    def test_pi4_address_does_not_hit_pi3_gpio(self):
        # Firmware built for the Pi 4 must not silently work on a Pi 3.
        board = run(RaspberryPi3(), """
            ldr r0, =0xFE200000
            mov r1, #1
            lsl r1, r1, #21
            str r1, [r0, #0x04]
        """)
        assert board.gpio.function(17) == GpioFunction.INPUT

    def test_bit_banged_pulse_train(self):
        board = run(RaspberryPi4(), """
            ldr r0, =0xFE200000
            mov r1, #1
            lsl r1, r1, #21
            str r1, [r0, #0x04]        @ pin 17 output
            mov r2, #1
            lsl r2, r2, #17
            mov r3, #4                 @ four pulses
        loop:
            str r2, [r0, #0x1C]        @ high
            str r2, [r0, #0x28]        @ low
            subs r3, r3, #1
            bne loop
        """)
        assert board.gpio.pulse_count(17) == 4
        levels = [lvl for _, lvl in board.gpio.transitions(17)]
        assert levels == [True, False] * 4

    def test_firmware_reads_input_level(self):
        board = RaspberryPi4()
        board.gpio.drive_input(4, True)
        run(board, """
            ldr r0, =0xFE200000
            ldr r1, [r0, #0x34]        @ GPLEV0
            and r1, r1, #0x10          @ isolate pin 4
        """)
        assert board.cpu.registers.get(1) == 0x10

    def test_trace_records_driver_register_sequence(self):
        board = run(RaspberryPi4(trace=True), """
            ldr r0, =0xFE200000
            mov r1, #1
            lsl r1, r1, #21
            str r1, [r0, #0x04]
            mov r2, #1
            lsl r2, r2, #17
            str r2, [r0, #0x1C]
        """)
        names = [a.name for a in board.gpio.accesses]
        assert names == ['GPFSEL1', 'GPSET0']
