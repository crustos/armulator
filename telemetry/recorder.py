"""
Sampling motor state over time so it can be plotted or asserted on.

The motor models are deliberately time-agnostic: :meth:`MotorHat.advance` takes a
duration and nothing anywhere accumulates it. That is the right call for the models --
the emulator counts instructions, not seconds -- but it means there is no time axis to
plot against. This module adds one, and owns it, without changing the models.

WHAT IT SAMPLES
---------------
Everything the motor stack exposes as a property: bridge mode and duty, signed drive,
and then whatever the attached load has -- shaft position and speed for a DC motor,
step counts for a stepper. A channel with nothing plugged into it still records its
bridge, because a duty cycle going to a position with no motor on it is exactly the
kind of driver bug worth seeing.

WHEN IT SAMPLES
---------------
Two triggers, and both matter:

*Time.* Attaching wraps :meth:`MotorHat.advance` so that a call asking for half a
second is chopped into ``interval``-sized slices, with a sample after each. Without
this, ``hat.advance(0.5)`` yields a single data point and the spin-up curve -- the
whole reason to plot a DC motor -- is invisible. Chopping is safe here because
:meth:`DcMotor.advance` integrates in closed form, so advancing by many small steps
gives the same answer as one large one. That is a property of the model, not an
approximation this module is making.

*Change.* Firmware writing a PWM register changes duty instantly. Sampled on a time
grid alone, an edge lands somewhere between two points and plots as a ramp, implying
a gradual change that never happened. So the recorder also subscribes to the PCA9685
and, on any output change, writes the *old* values at the current timestamp before
writing the new ones. Two rows share a timestamp and the edge plots vertical, which
is what the hardware did.

USAGE
-----
    hat = MotorHat().attach_to(board)
    motor = hat.attach_dc_motor(1)
    telemetry = MotorTelemetry(hat)

    board.run(2000)
    hat.advance(0.5)

    telemetry.series(1, 'speed')       # (times, values)

Manual sampling is always available regardless of the automatic hooks::

    telemetry.sample()

which is what you want when something outside ``advance`` moved the world, or when
taking a reading at a precise moment matters more than the grid.
"""

from armulator.peripherals.motor import BRAKE, COAST, FORWARD, REVERSE, DcMotor, StepperMotor

#: Bridge modes as numbers, so mode can be plotted on an axis. The ordering is chosen
#: to read sensibly on a chart: reverse below zero, coast at zero, forward above it.
#: Brake sits above forward because it is not a point on that continuum at all -- it is
#: a different thing the bridge is doing, and separating it visually is the point.
MODE_CODES = {REVERSE: -1, COAST: 0, FORWARD: 1, BRAKE: 2}

#: Signals recorded for every motor position, load or no load.
BRIDGE_FIELDS = ('duty', 'drive', 'mode_code', 'braking')

#: Extra signals for a DC motor.
DC_FIELDS = ('position', 'speed', 'angle', 'stalled')

#: Extra signals for a stepper.
STEPPER_FIELDS = ('steps', 'missed_steps', 'position', 'angle')


class MotorTelemetry:
    """
    Records the state of a :class:`~armulator.peripherals.motor_hat.MotorHat` over time.

    :param hat: the HAT to watch
    :param interval: sampling period in seconds for time-driven samples
    :param channels: motor positions to record; all four by default
    :param auto: wrap ``advance`` and subscribe to the controller on construction

    With ``auto=False`` nothing is hooked and :meth:`sample` is the only way rows get
    added -- useful when the caller wants full control of the time axis, or when
    wrapping ``advance`` would interfere with something else that already did.
    """

    def __init__(self, hat, interval=0.005, channels=None, auto=True):
        if interval <= 0:
            raise ValueError(f'interval must be positive, not {interval}')
        self.hat = hat
        self.interval = interval
        self.channels = tuple(sorted(channels if channels is not None else hat.channels))
        for index in self.channels:
            if index not in hat.channels:
                raise ValueError(f'{hat!r} has no motor position {index}')

        #: Seconds elapsed since attachment. The recorder owns this; nothing in the
        #: motor models tracks it.
        self.time = 0.0
        #: Timestamp of every row, ascending but not strictly -- edges duplicate one.
        self.times = []
        #: Why each row exists: 'initial', 'interval', 'edge' or 'manual'.
        self.reasons = []
        #: {channel index: {field name: [values]}}
        self.data = {
            index: {field: [] for field in self._fields_for(index)}
            for index in self.channels
        }
        #: Bridge mode as a string per row, alongside the numeric mode_code.
        self.modes = {index: [] for index in self.channels}

        self._attached = False
        self._original_advance = None
        if auto:
            self.attach()
        else:
            self.sample(reason='initial')

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def attach(self):
        """Wrap ``advance`` and subscribe to the PWM controller."""
        if self._attached:
            return self
        self._original_advance = self.hat.advance
        self.hat.advance = self._advance
        self._subscribe()
        self._attached = True
        self.sample(reason='initial')
        return self

    def detach(self):
        """
        Put ``advance`` back and stop recording.

        Data already collected is kept; this only stops more arriving.
        """
        if not self._attached:
            return self
        self.hat.advance = self._original_advance
        self._original_advance = None
        listeners = getattr(self.hat.controller, '_listeners', None)
        if listeners is not None and self._on_output_change in listeners:
            listeners.remove(self._on_output_change)
        self._attached = False
        return self

    def _subscribe(self):
        """
        Register the change listener, if it is not already registered.

        :meth:`Pca9685.reset` clears its listener list, and firmware resetting the
        controller mid-run is normal. Re-checking on every advance is cheap and means
        a reset costs at most one interval of edge resolution rather than silently
        deafening the recorder for the rest of the run.
        """
        listeners = getattr(self.hat.controller, '_listeners', None)
        if listeners is None or self._on_output_change not in listeners:
            self.hat.controller.on_change(self._on_output_change)

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------

    def _advance(self, seconds):
        """
        Stand-in for :meth:`MotorHat.advance` that samples as it goes.
        """
        if seconds <= 0:
            return self._original_advance(seconds)

        self._subscribe()
        remaining = seconds
        while remaining > 1e-12:
            step = min(self.interval, remaining)
            self._original_advance(step)
            self.time += step
            remaining -= step
            self._append(reason='interval')

    def _on_output_change(self, channel, duty):
        """
        A PWM output moved.

        The HAT's own listener runs first and has already pushed the new duty into the
        bridge, so ``self`` is looking at post-change state. Repeating the previous row
        at this timestamp first is what keeps the edge vertical instead of sloped.
        """
        if self.times:
            self._repeat_last()
        self._append(reason='edge')

    def sample(self, reason='manual'):
        """Take a reading now, at the current time."""
        self._append(reason=reason)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _fields_for(self, index):
        load = self.hat.channels[index].load
        if isinstance(load, DcMotor):
            return BRIDGE_FIELDS + DC_FIELDS
        if isinstance(load, StepperMotor):
            return BRIDGE_FIELDS + STEPPER_FIELDS
        return BRIDGE_FIELDS

    def _append(self, reason):
        self.times.append(self.time)
        self.reasons.append(reason)
        for index in self.channels:
            channel = self.hat.channels[index]
            bridge = channel.bridge
            load = channel.load
            self.modes[index].append(bridge.mode)
            series = self.data[index]
            for field in series:
                if field == 'mode_code':
                    value = MODE_CODES[bridge.mode]
                elif field in BRIDGE_FIELDS:
                    value = getattr(bridge, field)
                else:
                    value = getattr(load, field)
                series[field].append(float(value))

    def _repeat_last(self):
        """Duplicate the most recent row at the current timestamp."""
        self.times.append(self.time)
        self.reasons.append('edge')
        for index in self.channels:
            self.modes[index].append(self.modes[index][-1])
            for values in self.data[index].values():
                values.append(values[-1])

    # ------------------------------------------------------------------
    # Reading back
    # ------------------------------------------------------------------

    def series(self, index, field):
        """
        One signal as ``(times, values)``, ready to hand to matplotlib.

        :raises KeyError: if the channel was not recorded, or the load has no such field
        """
        if index not in self.data:
            raise KeyError(f'motor position {index} was not recorded')
        if field not in self.data[index]:
            available = ', '.join(sorted(self.data[index]))
            raise KeyError(
                f'M{index} has no signal {field!r}; recorded: {available}'
            )
        return list(self.times), list(self.data[index][field])

    def fields(self, index):
        """Which signals exist for a motor position."""
        return tuple(sorted(self.data[index]))

    @property
    def active_channels(self):
        """Positions whose bridge did something other than sit at coast."""
        return tuple(
            index for index in self.channels
            if any(code != MODE_CODES[COAST] for code in self.data[index]['mode_code'])
        )

    def __len__(self):
        return len(self.times)

    def __repr__(self):
        return (f'<MotorTelemetry {len(self.times)} samples '
                f'over {self.time:.3f}s, M{list(self.channels)}>')
