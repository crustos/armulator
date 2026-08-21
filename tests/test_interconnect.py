import pytest

from armulator.boards import JetsonNano, RaspberryPi3, RaspberryPi4
from armulator.boards.firmware import HAVE_KEYSTONE, firmware
from armulator.boards.interconnect import GpioLink, Machine, SpiBridge
from armulator.peripherals.gpio_bcm import GpioFunction

needs_keystone = pytest.mark.skipif(
    not HAVE_KEYSTONE, reason='keystone-engine required to assemble firmware'
)

PI_GPIO = 0xFE200000
PI_SPI = 0xFE204000
NANO_GPIO = 0x6000D000


def loaded(board, source):
    board.load(board.CODE_BASE, firmware(source, address=board.CODE_BASE))
    board.start()
    return board


def pi_drives_pin_high(pin=17, shift=21):
    return f"""
        ldr r0, ={PI_GPIO:#x}
        mov r1, #1
        lsl r1, r1, #{shift}
        str r1, [r0, #0x04]
        mov r2, #1
        lsl r2, r2, #{pin}
        str r2, [r0, #0x1C]
    """


class TestGpioLink:

    def test_output_propagates_to_receiver(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        pi.gpio.write_register(0x04, 0b001 << 21)      # pin 17 output
        pi.gpio.write_register(0x1C, 1 << 17)
        link = GpioLink(pi, 17, nano, 'PA0')
        link.settle()
        assert nano.gpio.level('PA0') is True

    def test_input_pin_does_not_drive_the_wire(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        # Pin 17 left as an input: the wire must float, not read as low.
        nano.gpio.drive_input('PA0', True)
        link = GpioLink(pi, 17, nano, 'PA0')
        link.settle()
        assert nano.gpio.level('PA0') is False         # released to its pull

    def test_link_follows_driver_changes(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        pi.gpio.write_register(0x04, 0b001 << 21)
        link = GpioLink(pi, 17, nano, 'PA0')
        link.settle()
        assert nano.gpio.level('PA0') is False
        pi.gpio.write_register(0x1C, 1 << 17)
        link.settle()
        assert nano.gpio.level('PA0') is True
        pi.gpio.write_register(0x28, 1 << 17)          # GPCLR0
        link.settle()
        assert nano.gpio.level('PA0') is False

    def test_inverting_link(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        pi.gpio.write_register(0x04, 0b001 << 21)
        link = GpioLink(pi, 17, nano, 'PA0', inverting=True)
        link.settle()
        assert nano.gpio.level('PA0') is True           # low driven -> high

    def test_edges_are_recorded(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        pi.gpio.write_register(0x04, 0b001 << 21)
        link = GpioLink(pi, 17, nano, 'PA0')
        link.settle(0)
        pi.gpio.write_register(0x1C, 1 << 17)
        link.settle(1)
        pi.gpio.write_register(0x28, 1 << 17)
        link.settle(2)
        assert [level for _, level in link.edges()] == [False, True, False]

    def test_tegra_can_drive_a_pi(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        nano.gpio.write_register(0x00, 1)              # PA0 GPIO
        nano.gpio.write_register(0x10, 1)              # output
        nano.gpio.write_register(0x20, 1)              # high
        link = GpioLink(nano, 'PA0', pi, 27)
        link.settle()
        assert pi.gpio.level(27) is True

    def test_pi_to_pi_link(self):
        a, b = RaspberryPi4(), RaspberryPi3()
        a.gpio.write_register(0x04, 0b001 << 21)
        a.gpio.write_register(0x1C, 1 << 17)
        link = GpioLink(a, 17, b, 5)
        link.settle()
        assert b.gpio.level(5) is True


class TestSpiBridge:

    def test_master_bytes_reach_the_slave_board(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        bridge = SpiBridge(pi, nano)
        pi.spi.write_register(0x04, 0xAB)
        assert [m for m, _ in bridge.exchanges] == [0xAB]

    def test_slave_response_returns_to_master(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        bridge = SpiBridge(pi, nano)
        bridge.queue_slave_response(b'\x5A')
        pi.spi.write_register(0x04, 0x00)
        assert pi.spi.read_register(0x04) == 0x5A

    def test_bridge_registers_on_the_right_chip_select(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        SpiBridge(pi, nano, chip_select=1)
        assert 1 in pi.spi.slaves
        assert 0 not in pi.spi.slaves


class TestMachine:

    def test_boards_are_accessible_by_name(self):
        machine = Machine()
        pi = machine.add('pi', RaspberryPi4())
        assert machine.pi is pi
        assert machine.boards['pi'] is pi

    def test_settle_propagates_all_links(self):
        pi, nano = RaspberryPi4(), JetsonNano()
        pi.gpio.write_register(0x04, 0b001 << 21)
        pi.gpio.write_register(0x1C, 1 << 17)
        machine = Machine()
        machine.add('pi', pi)
        machine.add('nano', nano)
        machine.link(GpioLink(pi, 17, nano, 'PA0'))
        machine.settle()
        assert nano.gpio.level('PA0') is True

    def test_run_until_stops_on_predicate(self):
        machine = Machine()
        pi = machine.add('pi', loaded(RaspberryPi4(), pi_drives_pin_high()))
        satisfied = machine.run_until(lambda: pi.gpio.level(17), 2000)
        assert satisfied is True

    def test_run_until_gives_up_within_budget(self):
        machine = Machine()
        machine.add('pi', loaded(RaspberryPi4(), 'mov r0, #1'))
        assert machine.run_until(lambda: False, 200) is False

    def test_run_stops_once_boards_halt(self):
        machine = Machine()
        machine.add('pi', loaded(RaspberryPi4(), 'mov r0, #1'))
        executed = machine.run(100000)
        assert executed < 100000        # halted rather than burning the budget


@needs_keystone
class TestCrossDeviceFirmware:

    def test_pi_signal_observed_by_jetson_firmware(self):
        pi = loaded(RaspberryPi4(), pi_drives_pin_high())
        nano = loaded(JetsonNano(), f"""
            ldr r0, ={NANO_GPIO:#x}
            mov r1, #1
            str r1, [r0, #0x00]        @ PA0 GPIO, input
        wait:
            ldr r2, [r0, #0x30]        @ IN
            tst r2, #1
            beq wait
            mov r3, #0xAA              @ marker
        """)
        machine = Machine(slice_size=8)
        machine.add('pi', pi)
        machine.add('nano', nano)
        machine.link(GpioLink(pi, 17, nano, 'PA0'))
        saw_it = machine.run_until(
            lambda: nano.cpu.registers.get(3) == 0xAA, 20000
        )
        assert saw_it, 'Jetson firmware never observed the Pi signal'

    def test_bidirectional_handshake_completes(self):
        pi = loaded(RaspberryPi4(), f"""
            ldr r0, ={PI_GPIO:#x}
            mov r1, #1
            lsl r1, r1, #21
            str r1, [r0, #0x04]
            mov r2, #1
            lsl r2, r2, #17
            str r2, [r0, #0x1C]        @ raise DATA
        wait_ready:
            ldr r3, [r0, #0x34]
            tst r3, #(1 << 27)
            beq wait_ready
            mov r4, #1
        """)
        nano = loaded(JetsonNano(), f"""
            ldr r0, ={NANO_GPIO:#x}
            mov r1, #3
            str r1, [r0, #0x00]
            mov r1, #2
            str r1, [r0, #0x10]        @ PA1 output, PA0 input
        wait_data:
            ldr r2, [r0, #0x30]
            tst r2, #1
            beq wait_data
            mov r3, #2
            str r3, [r0, #0x20]        @ raise READY
        """)
        machine = Machine(slice_size=8)
        machine.add('pi', pi)
        machine.add('nano', nano)
        machine.link(GpioLink(pi, 17, nano, 'PA0'))
        machine.link(GpioLink(nano, 'PA1', pi, 27))
        done = machine.run_until(
            lambda: pi.cpu.registers.get(4) == 1, 30000
        )
        assert done, 'handshake did not complete'
        assert nano.gpio.level('PA1') is True

    def test_handshake_stalls_without_the_return_wire(self):
        # Same firmware, but READY is not connected: the Pi must hang.
        pi = loaded(RaspberryPi4(), f"""
            ldr r0, ={PI_GPIO:#x}
            mov r1, #1
            lsl r1, r1, #21
            str r1, [r0, #0x04]
            mov r2, #1
            lsl r2, r2, #17
            str r2, [r0, #0x1C]
        wait_ready:
            ldr r3, [r0, #0x34]
            tst r3, #(1 << 27)
            beq wait_ready
            mov r4, #1
        """)
        nano = loaded(JetsonNano(), f"""
            ldr r0, ={NANO_GPIO:#x}
            mov r1, #3
            str r1, [r0, #0x00]
            mov r1, #2
            str r1, [r0, #0x10]
        wait_data:
            ldr r2, [r0, #0x30]
            tst r2, #1
            beq wait_data
            mov r3, #2
            str r3, [r0, #0x20]
        """)
        machine = Machine(slice_size=8)
        machine.add('pi', pi)
        machine.add('nano', nano)
        machine.link(GpioLink(pi, 17, nano, 'PA0'))     # DATA only
        done = machine.run_until(lambda: pi.cpu.registers.get(4) == 1, 5000)
        assert done is False
        assert nano.gpio.level('PA1') is True           # Jetson still replied

    def test_spi_payload_crosses_boards(self):
        payload = b'\xDE\xAD\xBE\xEF'
        body = f"""
            ldr r0, ={PI_SPI:#x}
            mov r1, #0x80
            str r1, [r0, #0x00]
        """
        for byte in payload:
            body += f"""
            mov r2, #{byte}
            str r2, [r0, #0x04]
        """
        pi = loaded(RaspberryPi4(), body)
        nano = loaded(JetsonNano(), 'mov r0, #0')
        machine = Machine()
        machine.add('pi', pi)
        machine.add('nano', nano)
        bridge = SpiBridge(pi, nano)
        machine.run(5000)
        assert bytes(m for m, _ in bridge.exchanges) == payload
