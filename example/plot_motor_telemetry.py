"""
Record a DC motor through a drive profile and plot it.

Runs without a board or firmware: the PCA9685 is programmed directly, the way firmware
would, so the example is about the telemetry rather than about getting a Pi booted. The
same recorder works unchanged when the register writes come from emulated code.

    python3 example/plot_motor_telemetry.py

Needs matplotlib:  python3 -m pip install armulator[plot]
"""

from armulator.peripherals.motor_hat import MotorHat
from armulator.peripherals.pca9685 import LED0_ON_L, MODE1, MODE1_AI
from armulator.telemetry import MotorTelemetry
from armulator.telemetry.plot import plot_hat, plot_motor


def set_channel(hat, channel, duty):
    """Program one PWM channel: four registers from a zero start, as a driver does."""
    count = int(duty * 4095)
    hat.controller.write([LED0_ON_L + 4 * channel, 0, 0, count & 0xFF, count >> 8])


def set_motor(hat, index, drive):
    """
    Drive a motor position at a signed duty, or brake it with ``drive=None``.

    Note the channel order per position comes from the HAT's schematic and is not
    uniform across the four -- MOTOR_CHANNELS is the authority, not a pattern.
    """
    pwm, in2, in1 = hat.channels[index].channels
    if drive is None:
        set_channel(hat, in1, 1.0)
        set_channel(hat, in2, 1.0)
        return
    set_channel(hat, in1, 1.0 if drive > 0 else 0.0)
    set_channel(hat, in2, 1.0 if drive < 0 else 0.0)
    set_channel(hat, pwm, abs(drive))


def main():
    hat = MotorHat()
    hat.controller.write([MODE1, MODE1_AI])      # awake, auto-increment on
    hat.attach_dc_motor(1, free_speed=2.0, spin_up=0.15)
    hat.attach_dc_motor(3, free_speed=3.0, spin_up=0.25)

    telemetry = MotorTelemetry(hat, interval=0.002)

    # A profile with something to look at: ramp up, hold, reverse, a duty too low to
    # overcome friction, then a hard brake.
    set_motor(hat, 1, 0.4)
    set_motor(hat, 3, 1.0)
    hat.advance(0.4)

    set_motor(hat, 1, 1.0)
    hat.advance(0.4)

    set_motor(hat, 1, -0.8)
    hat.advance(0.5)

    set_motor(hat, 1, 0.05)                      # below stall_drive: nothing happens
    hat.advance(0.3)

    set_motor(hat, 1, None)                      # brake
    set_motor(hat, 3, 0.0)                       # coast, for the contrast
    hat.advance(0.4)

    print(hat.format_state())
    print(telemetry)

    figure, _ = plot_motor(telemetry, 1, fields=('mode_code', 'drive', 'speed', 'position'))
    figure.savefig('motor_m1.png', dpi=120)

    figure, _ = plot_hat(telemetry, fields=('drive', 'speed'))
    figure.savefig('motor_hat.png', dpi=120)

    print('wrote motor_m1.png and motor_hat.png')


if __name__ == '__main__':
    main()
