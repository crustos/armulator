"""
The Adafruit DC and Stepper Motor HAT, and the wiring that makes it work.

A HAT is not a board — it plugs onto one. So this is not a :class:`~armulator.boards.Board`
subclass; it is an object that attaches itself to an existing Pi:

    board = RaspberryPi4()
    hat = MotorHat()
    hat.attach_to(board)

Attaching puts a PCA9685 on the Pi's I2C bus at 0x60 and subscribes four H-bridges to the
PWM channels the HAT routes to them. From then on, firmware writing I2C registers turns
shafts.

The channel map is the board designer's decision and nothing in the chip or the driver
implies it — it comes from the HAT's schematic, and getting it wrong is the classic
mistake when writing a driver from scratch. It is stated here explicitly:

    ======  =====  =====  =====
    Motor   PWM    IN2    IN1
    ======  =====  =====  =====
    M1      8      9      10
    M2      13     12     11
    M3      2      3      4
    M4      7      6      5
    ======  =====  =====  =====

Note that M2 and M4 have their direction channels in the opposite order to M1 and M3.
That is not a typo: it falls out of the PCB routing, and a driver that assumes a uniform
pattern drives two of the four motors backwards.

Steppers use two motor channels each, because a bipolar stepper needs two H-bridges:
stepper 1 is M1 and M2, stepper 2 is M3 and M4.
"""

from armulator.peripherals.motor import DcMotor, HBridge, StepperMotor
from armulator.peripherals.pca9685 import Pca9685

#: Default I2C address. The HAT's solder jumpers move it between 0x60 and 0x7F, which is
#: how up to 32 HATs stack on one Pi.
DEFAULT_ADDRESS = 0x60

#: (pwm, in2, in1) channel per motor, straight off the schematic.
MOTOR_CHANNELS = {
    1: (8, 9, 10),
    2: (13, 12, 11),
    3: (2, 3, 4),
    4: (7, 6, 5),
}

#: Which motor channels each stepper occupies.
STEPPER_MOTORS = {1: (1, 2), 2: (3, 4)}


class MotorChannel:
    """
    One motor position on the HAT: an H-bridge fed by three PWM channels.

    The bridge does not know which channels it is on; this class is the wiring between
    the controller's channel numbers and the bridge's three inputs.
    """

    def __init__(self, index, pwm_channel, in2_channel, in1_channel, bridge=None):
        self.index = index
        self.pwm_channel = pwm_channel
        self.in2_channel = in2_channel
        self.in1_channel = in1_channel
        self.bridge = bridge or HBridge(name=f'M{index}')
        #: What is plugged into this position, if anything.
        self.load = None

    @property
    def channels(self):
        return (self.pwm_channel, self.in2_channel, self.in1_channel)

    def apply(self, channel, duty):
        """Route a changed PWM channel to the right bridge input."""
        if channel == self.pwm_channel:
            self.bridge.set_inputs(pwm=duty)
        elif channel == self.in1_channel:
            self.bridge.set_inputs(in1=duty)
        elif channel == self.in2_channel:
            self.bridge.set_inputs(in2=duty)
        else:
            return False
        return True

    def __repr__(self):
        return f'<MotorChannel M{self.index} {self.bridge.mode} {self.bridge.duty:.2f}>'


class MotorHat:
    """
    :param address: I2C address of the PCA9685
    :param trace: record every I2C register write for assertions

    Four motor positions are always present. Attaching a load to one is what makes it
    turn something:

        hat.attach_dc_motor(1)                  # a DC motor on M1
        hat.attach_stepper(1)                   # a stepper across M1 and M2
    """

    def __init__(self, address=DEFAULT_ADDRESS, name='motor_hat', trace=False):
        self.name = name
        self.controller = Pca9685(address=address, name=f'{name}_pwm')
        self.trace = trace
        self.channels = {
            index: MotorChannel(index, *MOTOR_CHANNELS[index])
            for index in MOTOR_CHANNELS
        }
        #: Loads by name, so a test can reach them without knowing the wiring.
        self.motors = {}
        self.steppers = {}
        self.board = None
        self.controller.on_change(self._channel_changed)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def attach_to(self, board, bus=None):
        """
        Plug the HAT onto ``board``, putting the controller on its I2C bus.
        """
        target = bus if bus is not None else getattr(board, 'i2c', None)
        if target is None:
            raise ValueError(
                f'{type(board).__name__} has no I2C controller to plug a HAT onto'
            )
        target.attach_slave(self.controller, self.controller.address)
        self.board = board
        return self

    def attach_dc_motor(self, index, **kwargs):
        """
        Put a DC motor on motor position ``index`` (1-4).
        """
        if index not in self.channels:
            raise ValueError(f'motor position must be 1-4, not {index}')
        motor = DcMotor(name=f'M{index}', **kwargs)
        # The motor brings its own bridge; the channel drives that one instead.
        self.channels[index].bridge = motor.bridge
        self.channels[index].load = motor
        self.motors[index] = motor
        return motor

    def attach_stepper(self, index, **kwargs):
        """
        Put a bipolar stepper across a pair of motor positions.

        Stepper 1 uses M1 and M2, stepper 2 uses M3 and M4 - the two H-bridges a bipolar
        stepper needs, one per coil.
        """
        if index not in STEPPER_MOTORS:
            raise ValueError(f'stepper must be 1 or 2, not {index}')
        first, second = STEPPER_MOTORS[index]
        stepper = StepperMotor(name=f'S{index}', **kwargs)
        self.channels[first].bridge = stepper.coil_a
        self.channels[second].bridge = stepper.coil_b
        self.channels[first].load = stepper
        self.channels[second].load = stepper
        self.steppers[index] = stepper
        return stepper

    # ------------------------------------------------------------------
    # Reacting to firmware
    # ------------------------------------------------------------------

    def _channel_changed(self, channel, duty):
        """
        A PWM output moved: work out which motor it belongs to and update it.
        """
        for motor_channel in self.channels.values():
            if not motor_channel.apply(channel, duty):
                continue
            load = motor_channel.load
            # A stepper responds to its field changing rather than to time, so it is
            # stepped here; a DC motor waits for advance().
            if isinstance(load, StepperMotor):
                load.update()

    def advance(self, seconds):
        """
        Let the mechanical side run for ``seconds``.

        Only the DC motors have anything to do here. Steppers move when the coils change,
        which has already happened by the time this is called.
        """
        for motor in self.motors.values():
            motor.advance(seconds)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def channel_state(self, index):
        """The bridge mode and duty for a motor position."""
        bridge = self.channels[index].bridge
        return bridge.mode, bridge.duty

    def format_state(self):
        lines = [f'{self.controller!r}']
        for index in sorted(self.channels):
            channel = self.channels[index]
            load = channel.load
            detail = ''
            if isinstance(load, DcMotor):
                detail = f'  pos={load.position:+.3f}rev speed={load.speed:+.3f}rev/s'
            elif isinstance(load, StepperMotor):
                detail = f'  steps={load.steps:+d} missed={load.missed_steps}'
            lines.append(f'  M{index}: {channel.bridge.mode:8} '
                         f'duty={channel.bridge.duty:.2f}{detail}')
        return '\n'.join(lines)

    def __repr__(self):
        return f'<MotorHat 0x{self.controller.address:02X} {len(self.motors)} motors>'
