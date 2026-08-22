"""
The motor HAT: a PCA9685 on I2C driving H-bridges driving motors.

The point of this stack is that a test can assert on *shaft position* rather than on
register values. Register-level tests are here too, because that is where driver bugs
start, but the ones worth reading are the end-to-end cases: firmware writes I2C
registers and a shaft turns a measurable distance in a particular direction.
"""

import pytest

from armulator.boards import RaspberryPi4
from armulator.boards.firmware import HAVE_KEYSTONE, firmware
from armulator.peripherals.motor import BRAKE, COAST, FORWARD, REVERSE, HBridge
from armulator.peripherals.motor_hat import MOTOR_CHANNELS, MotorHat
from armulator.peripherals.pca9685 import (
    ALL_LED_OFF_H,
    FULL_BIT,
    LED0_ON_L,
    MODE1,
    MODE1_AI,
    MODE1_SLEEP,
    MODE2,
    MODE2_INVRT,
    PRE_SCALE,
    Pca9685,
)

# BCM2711 I2C1 registers, for the firmware tests.
I2C_BASE = 0xFE804000
BSC_C, BSC_DLEN, BSC_A, BSC_FIFO = 0x00, 0x08, 0x0C, 0x10
C_ST, C_CLEAR, C_I2CEN = 1 << 7, 0b11 << 4, 1 << 15


@pytest.fixture
def controller():
    pwm = Pca9685()
    pwm.write([MODE1, MODE1_AI])          # awake, auto-increment on
    return pwm


def set_channel(pwm, channel, duty):
    """Program one channel the way a driver does: four registers from a zero start."""
    count = int(duty * 4095)
    pwm.write([LED0_ON_L + 4 * channel, 0, 0, count & 0xFF, count >> 8])


class TestPca9685Registers:
    def test_it_wakes_up_asleep(self):
        pwm = Pca9685()
        assert pwm.sleeping is True
        # A sleeping chip drives nothing, whatever the LED registers say.
        set_channel(pwm, 0, 1.0)
        assert pwm.duty_cycle(0) == 0.0

    def test_waking_enables_the_outputs(self, controller):
        set_channel(controller, 0, 1.0)
        assert controller.duty_cycle(0) == pytest.approx(1.0, abs=0.001)

    def test_duty_cycle_from_on_and_off_counts(self, controller):
        controller.write([LED0_ON_L, 0x00, 0x00, 0x00, 0x08])   # off at 2048
        assert controller.duty_cycle(0) == pytest.approx(0.5)

    def test_a_nonzero_on_count_shifts_the_pulse_without_changing_width(self, controller):
        # on=1024, off=3072 is still half the period, just phase shifted.
        controller.write([LED0_ON_L, 0x00, 0x04, 0x00, 0x0C])
        assert controller.duty_cycle(0) == pytest.approx(0.5)

    def test_full_on_beats_full_off(self, controller):
        # Both force bits set. Most people guess off wins; the datasheet says on does.
        controller.write([LED0_ON_L, 0, FULL_BIT, 0, FULL_BIT])
        assert controller.duty_cycle(0) == 1.0

    def test_invert_flips_every_output(self, controller):
        set_channel(controller, 0, 0.25)
        controller.write([MODE2, MODE2_INVRT])
        assert controller.duty_cycle(0) == pytest.approx(0.75, abs=0.001)

    def test_all_led_writes_every_channel(self, controller):
        set_channel(controller, 3, 1.0)
        controller.write([ALL_LED_OFF_H, FULL_BIT])
        assert all(controller.duty_cycle(c) == 0.0 for c in range(16))

    def test_reads_come_back_from_the_pointer(self, controller):
        controller.write([LED0_ON_L, 0x11, 0x22, 0x33, 0x44])
        controller.write([LED0_ON_L])
        assert controller.read(4) == bytes([0x11, 0x22, 0x33, 0x44])


class TestPca9685Traps:
    """The three things that most often go wrong in a real driver."""

    def test_prescale_is_ignored_while_awake(self, controller):
        before = controller.prescale
        controller.write([PRE_SCALE, 0x63])
        # No error, no effect - which is exactly why this is hard to spot.
        assert controller.prescale == before

    def test_prescale_takes_while_asleep(self, controller):
        controller.write([MODE1, MODE1_AI | MODE1_SLEEP])
        controller.write([PRE_SCALE, 0x63])
        controller.write([MODE1, MODE1_AI])
        assert controller.prescale == 0x63
        assert controller.frequency == pytest.approx(61.0, abs=0.5)

    def test_a_block_write_without_auto_increment_lands_in_one_register(self):
        pwm = Pca9685()
        pwm.write([MODE1, 0])                       # awake, AI off
        pwm.write([LED0_ON_L, 0x11, 0x22, 0x33, 0x44])
        assert pwm.registers[LED0_ON_L] == 0x44     # last value wins
        assert pwm.registers.get(LED0_ON_L + 1, 0) == 0

    def test_the_same_write_with_auto_increment_works(self, controller):
        controller.write([LED0_ON_L, 0x11, 0x22, 0x33, 0x44])
        assert [controller.registers[LED0_ON_L + i] for i in range(4)] == \
            [0x11, 0x22, 0x33, 0x44]

    def test_the_requested_frequency_is_only_approached(self, controller):
        # 25MHz / (4096 * (prescale+1)) cannot hit every value; 1600 becomes 1526.
        controller.write([MODE1, MODE1_AI | MODE1_SLEEP])
        controller.write([PRE_SCALE, controller.prescale_for(1600)])
        controller.write([MODE1, MODE1_AI])
        assert controller.frequency == pytest.approx(1526, abs=1)


class TestHBridge:
    @pytest.mark.parametrize('in1, in2, expected', [
        (0.0, 0.0, COAST),
        (1.0, 0.0, FORWARD),
        (0.0, 1.0, REVERSE),
        (1.0, 1.0, BRAKE),
    ])
    def test_the_truth_table(self, in1, in2, expected):
        bridge = HBridge()
        bridge.set_inputs(in1=in1, in2=in2, pwm=1.0)
        assert bridge.mode == expected

    def test_drive_is_signed_by_direction(self):
        bridge = HBridge()
        bridge.set_inputs(in1=1.0, in2=0.0, pwm=0.75)
        assert bridge.drive == pytest.approx(0.75)
        bridge.set_inputs(in1=0.0, in2=1.0)
        assert bridge.drive == pytest.approx(-0.75)

    def test_braking_ignores_the_pwm_input(self):
        bridge = HBridge()
        bridge.set_inputs(in1=1.0, in2=1.0, pwm=1.0)
        assert bridge.drive == 0.0
        assert bridge.braking is True


class TestDcMotor:
    def _driven(self, duty=1.0, reverse=False):
        hat = MotorHat()
        motor = hat.attach_dc_motor(1)
        motor.bridge.set_inputs(in1=0.0 if reverse else 1.0,
                                in2=1.0 if reverse else 0.0, pwm=duty)
        return hat, motor

    def test_it_turns_forward(self):
        hat, motor = self._driven()
        hat.advance(1.0)
        assert motor.position > 0
        assert motor.speed > 0

    def test_it_turns_the_other_way_in_reverse(self):
        hat, motor = self._driven(reverse=True)
        hat.advance(1.0)
        assert motor.position < 0

    def test_speed_settles_in_proportion_to_duty(self):
        hat, motor = self._driven(duty=0.5)
        hat.advance(5.0)
        assert motor.speed == pytest.approx(motor.free_speed * 0.5, rel=0.01)

    def test_advancing_in_pieces_matches_advancing_at_once(self):
        # The closed-form integration means the result does not depend on how the
        # caller chops up time, which keeps tests from being sensitive to step size.
        whole, motor_whole = self._driven()
        whole.advance(1.0)
        pieces, motor_pieces = self._driven()
        for _ in range(100):
            pieces.advance(0.01)
        assert motor_pieces.position == pytest.approx(motor_whole.position, rel=1e-6)

    def test_braking_stops_faster_than_coasting(self):
        hat, motor = self._driven()
        hat.advance(2.0)
        running = motor.speed

        motor.bridge.set_inputs(in1=0.0, in2=0.0)      # coast
        hat.advance(0.2)
        coasting = motor.speed

        motor.speed = running
        motor.bridge.set_inputs(in1=1.0, in2=1.0)      # brake
        hat.advance(0.2)
        braked = motor.speed

        assert braked < coasting
        assert braked == pytest.approx(0.0, abs=0.05)

    def test_a_duty_below_stall_does_not_move_it(self):
        hat, motor = self._driven(duty=0.02)
        hat.advance(1.0)
        assert motor.position == 0.0
        assert motor.stalled is True


class TestStepper:
    SEQUENCE = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def _stepper(self):
        hat = MotorHat()
        stepper = hat.attach_stepper(1)
        return hat, stepper

    def _step_to(self, stepper, index):
        a, b = self.SEQUENCE[index % 4]
        stepper.coil_a.set_inputs(pwm=1.0 if a else 0.0,
                                  in1=1.0 if a > 0 else 0.0, in2=1.0 if a < 0 else 0.0)
        stepper.coil_b.set_inputs(pwm=1.0 if b else 0.0,
                                  in1=1.0 if b > 0 else 0.0, in2=1.0 if b < 0 else 0.0)
        stepper.update()

    def test_stepping_forward_counts_up(self):
        _, stepper = self._stepper()
        for index in range(9):
            self._step_to(stepper, index)
        # The first energisation aligns the rotor without counting as a step.
        assert stepper.steps == 8
        assert stepper.missed_steps == 0

    def test_stepping_backward_counts_down(self):
        _, stepper = self._stepper()
        for index in range(5):
            self._step_to(stepper, index)
        for index in range(3, -1, -1):
            self._step_to(stepper, index)
        assert stepper.steps == 0

    def test_a_full_revolution(self):
        _, stepper = self._stepper()
        for index in range(stepper.steps_per_revolution + 1):
            self._step_to(stepper, index)
        assert stepper.steps == stepper.steps_per_revolution
        assert stepper.position == pytest.approx(1.0)

    def test_skipping_a_position_is_recorded_as_a_missed_step(self):
        # A driver stepping too fast, or getting the sequence wrong, makes the field
        # jump further than the rotor can follow.
        _, stepper = self._stepper()
        self._step_to(stepper, 0)
        self._step_to(stepper, 1)
        assert stepper.missed_steps == 0
        self._step_to(stepper, 3)
        assert stepper.missed_steps == 2
        assert stepper.steps == 1        # the rotor did not follow

    def test_de_energising_holds_position(self):
        _, stepper = self._stepper()
        for index in range(5):
            self._step_to(stepper, index)
        held = stepper.steps
        stepper.coil_a.set_inputs(pwm=0.0, in1=0.0, in2=0.0)
        stepper.coil_b.set_inputs(pwm=0.0, in1=0.0, in2=0.0)
        stepper.update()
        assert stepper.steps == held
        assert stepper.energised is False


class TestHatWiring:
    def test_the_channel_map_matches_the_schematic(self):
        # M2 and M4 have their direction channels in the opposite order to M1 and M3;
        # a driver assuming a uniform pattern drives half the motors backwards.
        assert MOTOR_CHANNELS[1] == (8, 9, 10)
        assert MOTOR_CHANNELS[2] == (13, 12, 11)
        assert MOTOR_CHANNELS[3] == (2, 3, 4)
        assert MOTOR_CHANNELS[4] == (7, 6, 5)

    def test_a_pwm_write_reaches_the_right_bridge(self):
        hat = MotorHat()
        hat.controller.write([MODE1, MODE1_AI])
        hat.attach_dc_motor(3)
        pwm_channel, in2, in1 = MOTOR_CHANNELS[3]
        set_channel(hat.controller, in1, 1.0)
        set_channel(hat.controller, pwm_channel, 1.0)
        assert hat.channel_state(3)[0] == FORWARD

    def test_motors_are_independent(self):
        hat = MotorHat()
        hat.controller.write([MODE1, MODE1_AI])
        first = hat.attach_dc_motor(1)
        second = hat.attach_dc_motor(2)
        for channel, duty in ((MOTOR_CHANNELS[1][2], 1.0), (MOTOR_CHANNELS[1][0], 1.0)):
            set_channel(hat.controller, channel, duty)
        hat.advance(1.0)
        assert first.position > 0
        assert second.position == 0.0

    def test_attaching_needs_an_i2c_bus(self):
        class Bare:
            pass
        with pytest.raises(ValueError, match='no I2C controller'):
            MotorHat().attach_to(Bare())


@pytest.mark.skipif(not HAVE_KEYSTONE,
                    reason='keystone-engine required to assemble firmware')
class TestFirmwareDrivesTheMotor:
    """
    The end-to-end case: ARM firmware on the Pi, over emulated I2C, turning a shaft.
    """

    def _i2c_write(self, register, *values):
        payload = ''.join(f'    mov r2, #{value}\n'
                          f'    str r2, [r0, #{BSC_FIFO}]\n'
                          for value in (register,) + values)
        return (f'    mov r2, #{C_CLEAR}\n'
                f'    str r2, [r0, #{BSC_C}]\n'
                f'    mov r2, #{1 + len(values)}\n'
                f'    str r2, [r0, #{BSC_DLEN}]\n'
                f'{payload}'
                f'    ldr r2, ={C_I2CEN | C_ST}\n'
                f'    str r2, [r0, #{BSC_C}]\n')

    def _run(self, body, budget=8000):
        board = RaspberryPi4()
        hat = MotorHat().attach_to(board)
        motor = hat.attach_dc_motor(1)
        source = (f'    ldr r0, ={I2C_BASE}\n'
                  f'    mov r1, #0x60\n'
                  f'    str r1, [r0, #{BSC_A}]\n') + body
        board.load(board.CODE_BASE, firmware(source, address=board.CODE_BASE))
        board.start()
        board.run(budget)
        return board, hat, motor

    def _bring_up(self):
        return (self._i2c_write(MODE1, MODE1_SLEEP)
                + self._i2c_write(PRE_SCALE, 3)
                + self._i2c_write(MODE1, MODE1_AI))

    def _drive(self, channel, count):
        return self._i2c_write(LED0_ON_L + 4 * channel, 0, 0,
                               count & 0xFF, (count >> 8) & 0xFF)

    def test_firmware_reaches_the_controller(self):
        board, hat, _ = self._run(self._bring_up())
        assert board.halted is True
        assert board.i2c.target_address == 0x60
        assert hat.controller.sleeping is False
        assert hat.controller.frequency == pytest.approx(1526, abs=1)

    def test_firmware_turns_the_shaft_forward(self):
        body = (self._bring_up()
                + self._drive(10, 4095)      # IN1 high
                + self._drive(9, 0)          # IN2 low
                + self._drive(8, 2047))      # PWM at half
        _, hat, motor = self._run(body)
        assert hat.channel_state(1)[0] == FORWARD
        hat.advance(1.0)
        assert motor.position == pytest.approx(0.85, abs=0.05)

    def test_firmware_turns_the_shaft_in_reverse(self):
        body = (self._bring_up()
                + self._drive(10, 0)
                + self._drive(9, 4095)
                + self._drive(8, 4095))
        _, hat, motor = self._run(body)
        assert hat.channel_state(1)[0] == REVERSE
        hat.advance(1.0)
        assert motor.position < 0

    def test_firmware_that_forgets_to_wake_the_chip_drives_nothing(self):
        # The outputs are programmed correctly, but MODE1.SLEEP was never cleared.
        body = (self._i2c_write(MODE1, MODE1_SLEEP | MODE1_AI)
                + self._drive(10, 4095)
                + self._drive(8, 4095))
        _, hat, motor = self._run(body)
        assert hat.controller.sleeping is True
        hat.advance(1.0)
        assert motor.position == 0.0
