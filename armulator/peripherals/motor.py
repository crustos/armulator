"""
Motor drive: H-bridges and the motors they turn.

An H-bridge takes three logic inputs and turns them into a voltage across a motor. A DC
motor takes that voltage and turns it into rotation. Neither has any idea about I2C — the
PWM controller upstream is what connects them to firmware.

Keeping the layers separate is what makes the whole thing assertable. Firmware writes
PCA9685 registers; the H-bridge reads its three channels and works out a signed drive;
the motor integrates that into a position. A test can then say *the shaft turned this far
in this direction*, which is a claim about behaviour rather than about register values.

Time is explicit. The emulator counts instructions, not seconds, and the relationship
between them depends on a clock speed nothing here models. So a caller advances the
mechanical side by however long it decides has passed:

    board.run(2000)          # firmware programs the controller
    hat.advance(0.5)         # half a second of shaft time

That is honest about what is being simulated. Pretending instruction counts were seconds
would produce numbers that looked precise and meant nothing.
"""

import math

#: What an H-bridge is doing with its output.
COAST = 'coast'          # both outputs off, the motor freewheels
FORWARD = 'forward'
REVERSE = 'reverse'
BRAKE = 'brake'          # both outputs driven low, the motor is shorted and stops fast


class HBridge:
    """
    One half of a TB6612FNG, the driver on most Raspberry Pi motor HATs.

    Three inputs: two direction pins and a PWM enable. The direction pins decide what the
    bridge does and the PWM pin decides how hard:

    ====  ====  =========================================
    IN1   IN2   Result
    ====  ====  =========================================
    0     0     coast — outputs floating, motor freewheels
    1     0     forward
    0     1     reverse
    1     1     brake — outputs shorted, motor stops fast
    ====  ====  =========================================

    Coast and brake are genuinely different and the difference is worth reproducing: a
    coasting motor keeps spinning down under friction, a braked one stops sharply. Driver
    code that means to stop and only coasts is a common and hard-to-see bug.

    Inputs come from PWM channels, so they are duty cycles rather than logic levels. A
    channel used as a direction pin is driven fully on or fully off, and anything at or
    above :attr:`LOGIC_THRESHOLD` counts as high.
    """

    #: Duty cycle at or above which a direction input reads as logic high.
    LOGIC_THRESHOLD = 0.5

    def __init__(self, name='hbridge'):
        self.name = name
        self.in1 = 0.0
        self.in2 = 0.0
        self.pwm = 0.0
        #: Every state the bridge has been in, as (mode, duty), for assertions.
        self.history = []
        self._record()

    def set_inputs(self, in1=None, in2=None, pwm=None):
        if in1 is not None:
            self.in1 = in1
        if in2 is not None:
            self.in2 = in2
        if pwm is not None:
            self.pwm = pwm
        self._record()

    @property
    def mode(self):
        high1 = self.in1 >= self.LOGIC_THRESHOLD
        high2 = self.in2 >= self.LOGIC_THRESHOLD
        if high1 and high2:
            return BRAKE
        if high1:
            return FORWARD
        if high2:
            return REVERSE
        return COAST

    @property
    def duty(self):
        """How hard the bridge is driving, 0.0 to 1.0. Braking ignores the PWM pin."""
        if self.mode == BRAKE:
            return 0.0
        if self.mode == COAST:
            return 0.0
        return max(0.0, min(1.0, self.pwm))

    @property
    def drive(self):
        """
        Signed drive from -1.0 to 1.0: the fraction of supply across the motor.
        """
        if self.mode == FORWARD:
            return self.duty
        if self.mode == REVERSE:
            return -self.duty
        return 0.0

    @property
    def braking(self):
        return self.mode == BRAKE

    def _record(self):
        entry = (self.mode, self.duty)
        if not self.history or self.history[-1] != entry:
            self.history.append(entry)

    def __repr__(self):
        return f'<HBridge {self.name} {self.mode} {self.duty:.2f}>'


class DcMotor:
    """
    A brushed DC motor with a load.

    The model is first order: drive produces torque, friction opposes motion, and speed
    settles at a terminal value proportional to drive. That is enough to answer the
    questions a driver test actually asks — did it turn, which way, roughly how far, did
    it stop when told — without pretending to a fidelity no one has the parameters for.

    :param free_speed: shaft speed at full drive, in revolutions per second
    :param spin_up: time constant to reach that speed, in seconds
    :param stall_drive: drive below which the motor cannot overcome static friction
    """

    def __init__(self, free_speed=2.0, spin_up=0.15, stall_drive=0.08, name='motor'):
        self.name = name
        self.free_speed = free_speed
        self.spin_up = spin_up
        self.stall_drive = stall_drive
        #: Shaft position in revolutions. Signed and unbounded, so a test can assert on
        #: total travel rather than on an angle that wraps.
        self.position = 0.0
        #: Current speed in revolutions per second, signed.
        self.speed = 0.0
        self.bridge = HBridge(name=f'{name}_bridge')

    @property
    def angle(self):
        """Shaft angle in degrees, wrapped to 0-360."""
        return (self.position * 360.0) % 360.0

    @property
    def stalled(self):
        """
        True when the bridge is driving but not hard enough to move the shaft.

        Real motors do this, and a driver that sets a duty cycle too low to overcome
        friction sees nothing happen. Reporting it as stall rather than as motion is
        what lets a test catch that.
        """
        drive = self.bridge.drive
        return drive != 0.0 and abs(drive) < self.stall_drive and self.speed == 0.0

    def advance(self, dt):
        """
        Move the shaft forward by ``dt`` seconds.
        """
        if dt <= 0:
            return

        drive = self.bridge.drive
        if abs(drive) < self.stall_drive:
            target = 0.0
        else:
            target = drive * self.free_speed

        if self.bridge.braking:
            # A braked motor is shorted, so it stops much faster than it coasts.
            time_constant = self.spin_up / 4.0
            target = 0.0
        elif drive == 0.0:
            # Coasting: friction alone brings it down, which takes longer.
            time_constant = self.spin_up * 3.0
            target = 0.0
        else:
            time_constant = self.spin_up

        # Exponential approach to the target, integrated over the interval. Using the
        # closed form rather than stepping keeps the result independent of how the
        # caller chooses to chop up time.
        decay = math.exp(-dt / time_constant)
        start = self.speed
        self.speed = target + (start - target) * decay
        self.position += target * dt + (start - target) * time_constant * (1.0 - decay)

    def __repr__(self):
        return (f'<DcMotor {self.name} pos={self.position:.3f}rev '
                f'speed={self.speed:.3f}rev/s {self.bridge.mode}>')


class StepperMotor:
    """
    A bipolar stepper driven by two H-bridges, one per coil.

    Unlike a DC motor there is no speed to settle: the rotor follows the magnetic field
    the coils produce, and moves in discrete steps as that field rotates. What the model
    tracks is which way the field points and how many steps the rotor has taken.

    The interesting failure it reproduces is a **missed step**. If firmware advances the
    field by more than one position at a time — stepping too fast, or getting the
    sequence wrong — a real rotor cannot follow and slips. Here that is counted in
    :attr:`missed_steps` rather than silently applied, so a test can assert the commanded
    position and the actual one agree.

    :param steps_per_revolution: full steps per turn; 200 for a typical 1.8 degree motor
    """

    #: The four field positions of a full-step sequence, as (coil A, coil B) signs.
    FULL_STEP_SEQUENCE = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self, steps_per_revolution=200, name='stepper'):
        self.name = name
        self.steps_per_revolution = steps_per_revolution
        #: Net steps taken, signed.
        self.steps = 0
        #: Steps the rotor could not follow.
        self.missed_steps = 0
        self.coil_a = HBridge(name=f'{name}_a')
        self.coil_b = HBridge(name=f'{name}_b')
        self._field = None

    @property
    def position(self):
        """Shaft position in revolutions."""
        return self.steps / self.steps_per_revolution

    @property
    def angle(self):
        return (self.steps * 360.0 / self.steps_per_revolution) % 360.0

    @property
    def energised(self):
        return self.coil_a.drive != 0.0 or self.coil_b.drive != 0.0

    def _field_position(self):
        """
        Which of the four sequence positions the coils are currently producing.

        Returns None when the field is ambiguous - both coils off, or both driven, which
        is a half-step the full-step model has no position for.
        """
        a = self.coil_a.drive
        b = self.coil_b.drive
        if a == 0.0 and b == 0.0:
            return None
        if a != 0.0 and b != 0.0:
            return None
        signs = (0 if a == 0 else (1 if a > 0 else -1),
                 0 if b == 0 else (1 if b > 0 else -1))
        try:
            return self.FULL_STEP_SEQUENCE.index(signs)
        except ValueError:
            return None

    def update(self):
        """
        Re-read the coils and move the rotor to follow.

        Called whenever a coil input changes. A stepper responds to the field changing,
        not to time passing, which is why it has no ``advance``.
        """
        position = self._field_position()
        if position is None:
            return

        if self._field is None:
            # First energisation: the rotor snaps to whatever the field says without
            # that counting as a step.
            self._field = position
            return

        # Shortest way round the four positions.
        delta = (position - self._field) % 4
        if delta == 3:
            delta = -1

        if abs(delta) > 1:
            # The field jumped two positions at once; a real rotor cannot follow that
            # and slips instead of tracking it.
            self.missed_steps += abs(delta)
        else:
            self.steps += delta

        self._field = position

    def __repr__(self):
        return (f'<StepperMotor {self.name} steps={self.steps} '
                f'missed={self.missed_steps} pos={self.position:.3f}rev>')
