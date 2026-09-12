"""
Motor telemetry: turning a time-agnostic motor model into something plottable.

The claims worth testing are about the time axis, since that is the part the recorder
invents. Does a long advance produce a curve rather than two points? Does a register
write land as a vertical edge? Does subdividing change the physics? Those are the
questions; the plotting on top is a thin layer over the answers.
"""

import pytest

from armulator.peripherals.motor import BRAKE, COAST, FORWARD, REVERSE
from armulator.peripherals.motor_hat import MotorHat
from armulator.peripherals.pca9685 import LED0_ON_L, MODE1, MODE1_AI
from armulator.telemetry import MODE_CODES, MotorTelemetry


def set_channel(hat, channel, duty):
    """Program one PWM channel the way a driver does."""
    count = int(duty * 4095)
    hat.controller.write([LED0_ON_L + 4 * channel, 0, 0, count & 0xFF, count >> 8])


@pytest.fixture
def hat():
    hat = MotorHat()
    hat.controller.write([MODE1, MODE1_AI])
    return hat


def drive_forward(hat, index, duty=1.0):
    pwm, in2, in1 = hat.channels[index].channels
    set_channel(hat, in2, 0.0)
    set_channel(hat, in1, 1.0)
    set_channel(hat, pwm, duty)


class TestSampling:
    def test_it_samples_at_the_interval(self, hat):
        hat.attach_dc_motor(1)
        telemetry = MotorTelemetry(hat, interval=0.01)

        hat.advance(0.1)

        # One initial row plus ten slices; an unhooked advance would give one row.
        assert len(telemetry) == 11
        assert telemetry.time == pytest.approx(0.1)

    def test_a_long_advance_is_a_curve_not_two_points(self, hat):
        motor = hat.attach_dc_motor(1, free_speed=2.0, spin_up=0.15)
        telemetry = MotorTelemetry(hat, interval=0.005)
        drive_forward(hat, 1)

        hat.advance(0.5)

        _, speeds = telemetry.series(1, 'speed')
        # Strictly increasing spin-up, which is the shape that makes a plot worth having.
        assert len(speeds) > 50
        assert all(b >= a for a, b in zip(speeds, speeds[1:]))
        assert speeds[-1] == pytest.approx(motor.speed)

    def test_subdividing_does_not_change_the_physics(self, hat):
        """
        The recorder chops advance() into slices. That is only legitimate because
        DcMotor integrates in closed form, so it must give the same answer as one call.
        """
        plain = MotorHat()
        plain.controller.write([MODE1, MODE1_AI])
        reference = plain.attach_dc_motor(1)
        drive_forward(plain, 1)
        plain.advance(0.5)

        motor = hat.attach_dc_motor(1)
        MotorTelemetry(hat, interval=0.001)
        drive_forward(hat, 1)
        hat.advance(0.5)

        assert motor.position == pytest.approx(reference.position, rel=1e-9)
        assert motor.speed == pytest.approx(reference.speed, rel=1e-9)

    def test_manual_sampling_works_alongside_the_hooks(self, hat):
        hat.attach_dc_motor(1)
        telemetry = MotorTelemetry(hat, interval=0.01)

        hat.advance(0.02)
        before = len(telemetry)
        telemetry.sample()

        assert len(telemetry) == before + 1
        assert telemetry.reasons[-1] == 'manual'
        assert telemetry.times[-1] == pytest.approx(telemetry.times[-2])

    def test_manual_only_mode_takes_no_samples_on_its_own(self, hat):
        hat.attach_dc_motor(1)
        telemetry = MotorTelemetry(hat, auto=False)

        hat.advance(0.5)

        assert len(telemetry) == 1
        assert telemetry.time == 0.0


class TestEdges:
    def test_a_register_write_records_a_vertical_edge(self, hat):
        hat.attach_dc_motor(1)
        telemetry = MotorTelemetry(hat, interval=0.01)
        hat.advance(0.05)

        drive_forward(hat, 1, duty=0.75)

        times, drives = telemetry.series(1, 'drive')
        # The duplicated timestamp is what makes the step vertical rather than sloped.
        edges = [i for i in range(1, len(times)) if times[i] == times[i - 1]]
        assert edges
        index = edges[-1]
        assert drives[index - 1] != drives[index + 1] if index + 1 < len(drives) else True
        # Duty is quantised to the controller's 12 bits, so this is 0.7497, not 0.75.
        assert drives[-1] == pytest.approx(0.75, abs=1e-3)

    def test_it_survives_a_controller_reset(self, hat):
        """
        Pca9685.reset() drops its listeners. Firmware resetting the controller mid-run
        is normal, and the recorder has to keep working afterwards.
        """
        hat.attach_dc_motor(1)
        telemetry = MotorTelemetry(hat, interval=0.01)

        hat.controller.reset()
        hat.controller.write([MODE1, MODE1_AI])
        hat.advance(0.01)          # re-subscribes
        before = len(telemetry)
        drive_forward(hat, 1, duty=0.5)

        assert 'edge' in telemetry.reasons[before:]

    @pytest.mark.xfail(
        reason='pre-existing: Pca9685.reset() clears _listeners, which drops '
               'MotorHat._channel_changed too, so the HAT stops routing PWM changes '
               'to its bridges. Firmware that resets the controller mid-run leaves '
               'every motor dead while register writes still appear to succeed.',
        strict=True,
    )
    def test_the_hat_survives_a_controller_reset(self, hat):
        motor = hat.attach_dc_motor(1)

        hat.controller.reset()
        hat.controller.write([MODE1, MODE1_AI])
        drive_forward(hat, 1, duty=0.5)
        hat.advance(0.5)

        assert motor.bridge.drive == pytest.approx(0.5, abs=1e-3)
        assert motor.position > 0.0


class TestSignals:
    def test_it_records_direction_as_a_sign(self, hat):
        hat.attach_dc_motor(1)
        telemetry = MotorTelemetry(hat, interval=0.01)
        pwm, in2, in1 = hat.channels[1].channels

        set_channel(hat, in1, 1.0)
        set_channel(hat, pwm, 0.6)
        hat.advance(0.05)
        set_channel(hat, in1, 0.0)
        set_channel(hat, in2, 1.0)
        hat.advance(0.05)

        modes = telemetry.modes[1]
        assert FORWARD in modes and REVERSE in modes
        assert min(telemetry.data[1]['drive']) < 0 < max(telemetry.data[1]['drive'])

    def test_brake_is_distinguishable_from_coast(self, hat):
        hat.attach_dc_motor(1)
        telemetry = MotorTelemetry(hat, interval=0.01)
        pwm, in2, in1 = hat.channels[1].channels

        set_channel(hat, in1, 1.0)
        set_channel(hat, pwm, 1.0)
        hat.advance(0.05)
        set_channel(hat, in2, 1.0)          # both high: brake
        hat.advance(0.05)

        codes = telemetry.data[1]['mode_code']
        assert MODE_CODES[BRAKE] in codes
        assert MODE_CODES[COAST] in codes   # the initial state
        assert telemetry.data[1]['braking'][-1] == 1.0

    def test_a_stepper_records_step_counts(self, hat):
        stepper = hat.attach_stepper(1)
        telemetry = MotorTelemetry(hat, interval=0.01)

        assert 'steps' in telemetry.fields(1)
        assert 'speed' not in telemetry.fields(1)

        for coil, index in ((1, 1), (2, 1)):
            pwm, in2, in1 = hat.channels[coil].channels
            set_channel(hat, pwm, 1.0)
        telemetry.sample()
        assert telemetry.series(1, 'missed_steps')[1][-1] == stepper.missed_steps

    def test_an_empty_position_still_records_its_bridge(self, hat):
        telemetry = MotorTelemetry(hat, interval=0.01)
        drive_forward(hat, 2, duty=0.4)

        # Nothing is plugged into M2, but firmware driving it is worth seeing.
        assert telemetry.fields(2) == ('braking', 'drive', 'duty', 'mode_code')
        assert telemetry.series(2, 'duty')[1][-1] == pytest.approx(0.4, abs=1e-3)
        with pytest.raises(KeyError):
            telemetry.series(2, 'speed')

    def test_active_channels_ignores_idle_positions(self, hat):
        hat.attach_dc_motor(1)
        hat.attach_dc_motor(3)
        telemetry = MotorTelemetry(hat, interval=0.01)

        drive_forward(hat, 1)
        hat.advance(0.05)

        assert telemetry.active_channels == (1,)


class TestLifecycle:
    def test_detach_restores_advance(self, hat):
        hat.attach_dc_motor(1)
        original = MotorHat.advance
        telemetry = MotorTelemetry(hat, interval=0.01)
        assert hat.advance.__func__ is not original

        telemetry.detach()
        hat.advance(0.5)

        assert len(telemetry) == 1
        assert hat.advance.__func__ is original

    def test_it_rejects_a_position_the_hat_does_not_have(self, hat):
        with pytest.raises(ValueError):
            MotorTelemetry(hat, channels=[9])

    def test_it_rejects_a_nonsense_interval(self, hat):
        with pytest.raises(ValueError):
            MotorTelemetry(hat, interval=0)
