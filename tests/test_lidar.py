"""
The Lidar: a motor HAT position that spins a ranging head and talks RPLIDAR over UART.

The claims worth testing are the ones that cross a layer. Ray casting is checkable
against geometry anyone can do on paper. The protocol is checkable by round-tripping.
And the end-to-end case -- firmware writes PWM registers, firmware writes a scan
command, and measurement packets come back out of the UART -- is the one that says the
whole stack is wired up, which no single-layer test can.
"""

import math

import pytest

from armulator.peripherals.motor_hat import MotorHat
from armulator.peripherals.pca9685 import LED0_ON_L, MODE1, MODE1_AI
from armulator.peripherals.uart_pl011 import Pl011Uart
from armulator.sensors import (
    CMD_GET_HEALTH, CMD_SCAN, CMD_STOP, SYNC, Lidar, Measurement, Room,
    decode_measurement, encode_measurement,
)
from armulator.telemetry import MotorTelemetry


def set_channel(hat, channel, duty):
    count = int(duty * 4095)
    hat.controller.write([LED0_ON_L + 4 * channel, 0, 0, count & 0xFF, count >> 8])


def spin(hat, index, duty=1.0):
    pwm, in2, in1 = hat.channels[index].channels
    set_channel(hat, in2, 0.0)
    set_channel(hat, in1, 1.0)
    set_channel(hat, pwm, duty)


@pytest.fixture
def hat():
    hat = MotorHat()
    hat.controller.write([MODE1, MODE1_AI])
    return hat


@pytest.fixture
def uart():
    return Pl011Uart()


class TestGeometry:
    def test_a_centred_sensor_sees_the_wall_at_the_radius(self):
        room = Room(radius=4.0)
        for bearing in (0, 90, 180, 270):
            distance = room.range_at((0.0, 0.0), math.radians(bearing))
            assert distance == pytest.approx(4.0)

    def test_moving_off_centre_skews_the_wall(self):
        """The whole reason to ray-cast rather than use a table."""
        room = Room(radius=4.0)
        ahead = room.range_at((1.0, 0.0), 0.0)
        behind = room.range_at((1.0, 0.0), math.pi)

        assert ahead == pytest.approx(3.0)
        assert behind == pytest.approx(5.0)

    def test_an_obstacle_occludes_the_wall(self):
        room = Room(radius=4.0)
        room.add_obstacle(2.0, 0.0, 0.5)

        assert room.range_at((0.0, 0.0), 0.0) == pytest.approx(1.5)
        # A bearing that misses it still reaches the wall.
        assert room.range_at((0.0, 0.0), math.pi) == pytest.approx(4.0)

    def test_contains_knows_walls_and_obstacles(self):
        room = Room(radius=4.0)
        room.add_obstacle(2.0, 0.0, 0.5)

        assert room.contains(0.0, 0.0)
        assert not room.contains(5.0, 0.0)
        assert not room.contains(2.0, 0.0)

    def test_it_rejects_a_nonsense_room(self):
        with pytest.raises(ValueError):
            Room(radius=0)
        with pytest.raises(ValueError):
            Room().add_obstacle(0, 0, -1)


class TestProtocol:
    def test_a_measurement_round_trips(self):
        original = Measurement(angle=123.5, distance=1.75, start=True, quality=47)

        decoded = decode_measurement(encode_measurement(original))

        assert decoded.angle == pytest.approx(123.5, abs=0.02)
        assert decoded.distance == pytest.approx(1.75, abs=0.001)
        assert decoded.start is True
        assert decoded.quality == 47

    def test_no_return_encodes_as_zero_distance(self):
        node = encode_measurement(Measurement(10.0, None, False, 0))

        assert node[3] == 0 and node[4] == 0
        assert decode_measurement(node).distance is None

    def test_lost_framing_is_detectable(self):
        """
        The redundant start bit and check bit exist so a receiver can resync. A driver's
        resync path is worth being able to test, so they have to actually be wrong when
        framing is wrong.
        """
        node = bytearray(encode_measurement(Measurement(10.0, 1.0, True, 47)))
        node[0] |= 0b11                      # start and its inverse now agree
        with pytest.raises(ValueError):
            decode_measurement(bytes(node))

        node = bytearray(encode_measurement(Measurement(10.0, 1.0, True, 47)))
        node[1] &= 0xFE                      # check bit cleared
        with pytest.raises(ValueError):
            decode_measurement(bytes(node))


class TestSpinning:
    def test_it_does_not_scan_until_firmware_spins_it(self, hat):
        lidar = Lidar(room=Room(4.0)).attach_to(hat, 4)

        hat.advance(0.5)

        assert not lidar.scanning
        assert lidar.measurements == []

    def test_firmware_writing_pwm_makes_it_spin(self, hat):
        lidar = Lidar(room=Room(4.0)).attach_to(hat, 4)

        spin(hat, 4)
        hat.advance(1.0)

        assert lidar.scanning
        assert lidar.rpm > 100
        assert lidar.revolutions > 1
        assert len(lidar.measurements) > 1000

    def test_spinning_slower_gives_a_denser_scan(self, hat):
        """
        Sample rate is fixed and independent of spin speed, so points per revolution
        falls out of the ratio. A driver spinning too fast gets a sparse scan.
        """
        fast = Lidar(room=Room(4.0), free_speed=10.0).attach_to(hat, 4)
        spin(hat, 4)
        hat.advance(2.0)
        dense = fast.points_per_revolution

        slow_hat = MotorHat()
        slow_hat.controller.write([MODE1, MODE1_AI])
        slow = Lidar(room=Room(4.0), free_speed=2.0).attach_to(slow_hat, 4)
        spin(slow_hat, 4)
        slow_hat.advance(2.0)

        assert slow.points_per_revolution > dense

    def test_a_scan_traces_the_room(self, hat):
        lidar = Lidar(room=Room(4.0), max_range=6.0).attach_to(hat, 4)

        spin(hat, 4)
        hat.advance(2.0)

        angles, ranges = lidar.scan_points()
        assert len(angles) > 100
        assert max(angles) > 300 and min(angles) < 60
        # Centred in a circular room, every return is the wall radius.
        assert all(r == pytest.approx(4.0, abs=0.01) for r in ranges)

    def test_an_obstacle_shows_up_at_the_right_bearing(self, hat):
        room = Room(4.0)
        room.add_obstacle(0.0, 2.0, 0.4)     # due north of a centred sensor
        lidar = Lidar(room=room).attach_to(hat, 4)

        spin(hat, 4)
        hat.advance(2.0)

        angles, ranges = lidar.scan_points()
        near = [a for a, r in zip(angles, ranges) if r < 2.0]
        assert near
        assert all(abs(((a - 90.0 + 180) % 360) - 180) < 20 for a in near)

    def test_out_of_range_surfaces_return_nothing(self, hat):
        lidar = Lidar(room=Room(10.0), max_range=4.0).attach_to(hat, 4)

        spin(hat, 4)
        hat.advance(1.0)

        assert lidar.measurements
        assert all(m.distance is None for m in lidar.measurements)

    def test_it_rejects_a_position_the_hat_does_not_have(self, hat):
        with pytest.raises(ValueError):
            Lidar().attach_to(hat, 9)


class TestHostProtocol:
    def test_a_scan_command_starts_the_stream(self, hat, uart):
        lidar = Lidar(room=Room(4.0)).attach_to(hat, 4).connect(uart)
        spin(hat, 4)

        uart.tx_buffer.clear()
        for byte in (SYNC, CMD_SCAN):
            for callback in uart.tx_callbacks:
                callback(byte)
        hat.advance(0.5)

        assert CMD_SCAN in lidar.commands
        assert lidar.streaming
        # Descriptor then five-byte nodes.
        assert uart._rx_fifo[:2] == bytes([SYNC, 0x5A])
        assert len(uart._rx_fifo) > 7

    def test_stop_ends_the_stream(self, hat, uart):
        lidar = Lidar(room=Room(4.0)).attach_to(hat, 4).connect(uart)
        spin(hat, 4)
        for byte in (SYNC, CMD_SCAN):
            for callback in uart.tx_callbacks:
                callback(byte)
        hat.advance(0.2)

        for byte in (SYNC, CMD_STOP):
            for callback in uart.tx_callbacks:
                callback(byte)
        uart._rx_fifo.clear()
        hat.advance(0.2)

        assert not lidar.streaming
        assert len(uart._rx_fifo) == 0
        # It keeps ranging internally; it just stops telling anyone.
        assert lidar.measurements

    def test_health_query_answers(self, hat, uart):
        Lidar(room=Room(4.0)).attach_to(hat, 4).connect(uart)

        for byte in (SYNC, CMD_GET_HEALTH):
            for callback in uart.tx_callbacks:
                callback(byte)

        assert len(uart._rx_fifo) == 7 + 3
        assert uart._rx_fifo[-3] == 0        # good

    def test_bytes_before_a_sync_are_discarded(self, hat, uart):
        lidar = Lidar(room=Room(4.0)).attach_to(hat, 4).connect(uart)

        for byte in (0x00, 0xFF, 0x12, SYNC, CMD_SCAN):
            for callback in uart.tx_callbacks:
                callback(byte)

        assert lidar.commands == [CMD_SCAN]

    def test_streamed_nodes_decode_back(self, hat, uart):
        lidar = Lidar(room=Room(4.0)).attach_to(hat, 4).connect(uart)
        spin(hat, 4)
        for byte in (SYNC, CMD_SCAN):
            for callback in uart.tx_callbacks:
                callback(byte)
        uart._rx_fifo.clear()
        hat.advance(0.3)

        stream = bytes(uart._rx_fifo)
        nodes = [stream[i:i + 5] for i in range(0, len(stream) - 4, 5)]
        decoded = [decode_measurement(node) for node in nodes[:50]]

        assert decoded
        assert all(m.distance == pytest.approx(4.0, abs=0.01) for m in decoded)
        assert lidar.streaming


class TestTelemetry:
    def test_the_recorder_picks_up_lidar_signals(self, hat):
        lidar = Lidar(room=Room(4.0)).attach_to(hat, 4)
        telemetry = MotorTelemetry(hat, interval=0.01)

        spin(hat, 4)
        hat.advance(1.0)

        assert 'rpm' in telemetry.fields(4)
        assert 'angle' in telemetry.fields(4)
        times, rpm = telemetry.series(4, 'rpm')
        assert max(rpm) > 100
        # Shares the HAT's clock rather than running a second one.
        assert times[-1] == pytest.approx(1.0)
        assert telemetry.series(4, 'revolutions')[1][-1] == lidar.revolutions
