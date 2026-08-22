"""
NXP PCA9685 — 16-channel, 12-bit PWM controller on I2C.

This is the chip on almost every Raspberry Pi motor HAT and servo driver. It has no
motor logic of its own: it produces sixteen independent PWM outputs, and what those
outputs are wired to is the board designer's business. On a motor HAT they go to H-bridge
inputs; on a servo HAT they go straight to servo signal pins.

The model is faithful to the register interface, because that is where driver bugs live.
Three details in particular catch people out and are reproduced rather than smoothed
over:

* **The prescaler can only be written while the chip is asleep.** Firmware that sets the
  PWM frequency without first setting MODE1.SLEEP gets its write silently ignored, and
  then wonders why the servo twitches. Here the write is ignored too.

* **Auto-increment is off at reset.** A driver that writes all four LED registers in one
  I2C transaction, without setting MODE1.AI, writes four values into the *same* register.
  The model does exactly that.

* **Full-on beats full-off.** Bit 4 of the ON_H and OFF_H registers force a channel fully
  on or fully off regardless of the counts. If both are set, full-on wins - which is the
  opposite of what most people guess.

Outputs are read through :meth:`duty_cycle`, or subscribed to with :meth:`on_change`, so
whatever the channel drives can react when firmware reprograms it.
"""

#: Register map.
MODE1 = 0x00
MODE2 = 0x01
SUBADR1 = 0x02
SUBADR2 = 0x03
SUBADR3 = 0x04
ALLCALLADR = 0x05
LED0_ON_L = 0x06
ALL_LED_ON_L = 0xFA
ALL_LED_ON_H = 0xFB
ALL_LED_OFF_L = 0xFC
ALL_LED_OFF_H = 0xFD
PRE_SCALE = 0xFE
TESTMODE = 0xFF

#: MODE1 bits.
MODE1_ALLCALL = 1 << 0
MODE1_SUB3 = 1 << 1
MODE1_SUB2 = 1 << 2
MODE1_SUB1 = 1 << 3
MODE1_SLEEP = 1 << 4
MODE1_AI = 1 << 5
MODE1_EXTCLK = 1 << 6
MODE1_RESTART = 1 << 7

#: MODE2 bits.
MODE2_OUTNE0 = 1 << 0
MODE2_OUTNE1 = 1 << 1
MODE2_OUTDRV = 1 << 2
MODE2_OCH = 1 << 3
MODE2_INVRT = 1 << 4

#: Bit 4 of an ON_H or OFF_H register forces the channel fully on or fully off.
FULL_BIT = 1 << 4

#: The counter is 12 bits, so a period is 4096 steps.
PWM_STEPS = 4096

#: Internal oscillator, in Hz. The datasheet's frequency formula is built on it.
INTERNAL_CLOCK = 25_000_000

#: Reset values. MODE1 has SLEEP and ALLCALL set; the chip wakes up asleep.
RESET_MODE1 = MODE1_SLEEP | MODE1_ALLCALL
RESET_PRESCALE = 0x1E          # 200 Hz with the internal oscillator

CHANNELS = 16


class Pca9685:
    """
    :param address: 7-bit I2C address. The Adafruit motor HAT uses 0x60.
    :param name: label used in traces
    """

    def __init__(self, address=0x60, name='pca9685'):
        self.address = address
        self.name = name
        #: Every write the master made, as (register, value), for assertions.
        self.writes = []
        self.reset()

    def reset(self):
        self.registers = {MODE1: RESET_MODE1, MODE2: MODE2_OUTDRV,
                          PRE_SCALE: RESET_PRESCALE,
                          SUBADR1: 0xE2, SUBADR2: 0xE4, SUBADR3: 0xE8,
                          ALLCALLADR: 0xE0}
        self._pointer = 0
        self._listeners = []
        self.writes = []

    # ------------------------------------------------------------------
    # I2C slave interface
    # ------------------------------------------------------------------

    def write(self, data):
        """
        A write transaction: the first byte is the register pointer, the rest are values.

        Whether the pointer advances between values depends on MODE1.AI, which is the
        behaviour a driver has to opt into and frequently forgets to.
        """
        if not data:
            return
        self._pointer = data[0]
        auto_increment = bool(self.registers.get(MODE1, 0) & MODE1_AI)
        for value in data[1:]:
            self._write_register(self._pointer, value)
            if auto_increment:
                self._pointer = (self._pointer + 1) & 0xFF

    def read(self, count):
        """A read transaction, starting from the current pointer."""
        out = bytearray()
        pointer = self._pointer
        for _ in range(count):
            out.append(self.registers.get(pointer, 0) & 0xFF)
            pointer = (pointer + 1) & 0xFF
        return bytes(out)

    # ------------------------------------------------------------------
    # Register behaviour
    # ------------------------------------------------------------------

    def _write_register(self, register, value):
        value &= 0xFF
        self.writes.append((register, value))

        if register == PRE_SCALE and not self.sleeping:
            # The datasheet is explicit: PRE_SCALE is writable only in sleep. A driver
            # that skips the sleep gets no error, just no frequency change.
            return

        before = self._output_snapshot()
        self.registers[register] = value

        if register == MODE1 and not value & MODE1_SLEEP:
            # Coming out of sleep clears RESTART; the outputs resume.
            self.registers[MODE1] = value & ~MODE1_RESTART

        if register in (ALL_LED_ON_L, ALL_LED_ON_H, ALL_LED_OFF_L, ALL_LED_OFF_H):
            self._apply_all_led(register, value)

        self._notify(before)

    def _apply_all_led(self, register, value):
        """The ALL_LED registers write every channel at once."""
        offset = {ALL_LED_ON_L: 0, ALL_LED_ON_H: 1,
                  ALL_LED_OFF_L: 2, ALL_LED_OFF_H: 3}[register]
        for channel in range(CHANNELS):
            self.registers[LED0_ON_L + 4 * channel + offset] = value

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def sleeping(self):
        return bool(self.registers.get(MODE1, 0) & MODE1_SLEEP)

    @property
    def auto_increment(self):
        return bool(self.registers.get(MODE1, 0) & MODE1_AI)

    @property
    def inverted(self):
        return bool(self.registers.get(MODE2, 0) & MODE2_INVRT)

    @property
    def prescale(self):
        return self.registers.get(PRE_SCALE, RESET_PRESCALE)

    @property
    def frequency(self):
        """
        Output frequency in Hz, from the datasheet's formula.

        The rounding in the reverse direction is why asking for 1600 Hz and reading back
        1626 Hz is normal rather than a bug.
        """
        return INTERNAL_CLOCK / (PWM_STEPS * (self.prescale + 1))

    def prescale_for(self, frequency):
        """The PRE_SCALE value a driver should write for a target frequency."""
        return max(3, min(255, round(INTERNAL_CLOCK / (PWM_STEPS * frequency)) - 1))

    def channel_counts(self, channel):
        """The raw (on, off) counts for a channel, including the full-on/off bits."""
        base = LED0_ON_L + 4 * channel
        on = self.registers.get(base, 0) | (self.registers.get(base + 1, 0) << 8)
        off = self.registers.get(base + 2, 0) | (self.registers.get(base + 3, 0) << 8)
        return on, off

    def duty_cycle(self, channel):
        """
        The channel's duty cycle as a fraction from 0.0 to 1.0.

        A sleeping chip drives nothing, so every channel reads as zero - which is the
        state firmware leaves it in if it programs the outputs and forgets to wake up.
        """
        if self.sleeping:
            return 0.0

        base = LED0_ON_L + 4 * channel
        on_high = self.registers.get(base + 1, 0)
        off_high = self.registers.get(base + 3, 0)

        # Full-on takes precedence over full-off, which is the opposite of the usual
        # guess and a real source of stuck outputs.
        if on_high & FULL_BIT:
            duty = 1.0
        elif off_high & FULL_BIT:
            duty = 0.0
        else:
            on, off = self.channel_counts(channel)
            duty = ((off & 0x0FFF) - (on & 0x0FFF)) % PWM_STEPS / PWM_STEPS

        return 1.0 - duty if self.inverted else duty

    def outputs(self):
        """Every channel's duty cycle, for a quick look at the whole chip."""
        return [self.duty_cycle(channel) for channel in range(CHANNELS)]

    # ------------------------------------------------------------------
    # Change notification
    # ------------------------------------------------------------------

    def on_change(self, listener):
        """
        Register ``listener(channel, duty)``, called whenever an output changes.

        This is how whatever the channel drives finds out: an H-bridge subscribes to its
        three channels and re-evaluates when any of them moves.
        """
        self._listeners.append(listener)
        return listener

    def _output_snapshot(self):
        return self.outputs()

    def _notify(self, before):
        if not self._listeners:
            return
        after = self.outputs()
        for channel, (old, new) in enumerate(zip(before, after)):
            if old != new:
                for listener in self._listeners:
                    listener(channel, new)

    def __repr__(self):
        state = 'asleep' if self.sleeping else f'{self.frequency:.0f}Hz'
        return f'<Pca9685 0x{self.address:02X} {state}>'
