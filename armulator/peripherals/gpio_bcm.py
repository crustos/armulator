"""
Broadcom GPIO controller (BCM2835 / BCM2837 / BCM2711).

This is the peripheral behind ``/dev/gpiomem`` on the Raspberry Pi and the
one almost all Pi GPIO code drives.  The register block is identical across
BCM2835 (Pi 1), BCM2837 (Pi 3) and BCM2711 (Pi 4) except for the pull
up/down mechanism, which BCM2711 replaced -- see ``pull_style``.

Beyond emulating the registers, this model exposes a pin-level API so tests
can drive inputs and assert on outputs:

    gpio.drive_input(17, True)      # external world pulls pin 17 high
    gpio.level(17)                  # effective level on the pin
    gpio.function(17)               # GpioFunction.OUTPUT, ALT0, ...
    gpio.transitions(17)            # recorded output waveform
"""

from enum import IntEnum

from armulator.peripherals.mmio import MMIODevice

NUM_PINS = 54


class GpioFunction(IntEnum):
    """Values of the 3-bit FSEL field for a pin."""
    INPUT = 0b000
    OUTPUT = 0b001
    ALT5 = 0b010
    ALT4 = 0b011
    ALT0 = 0b100
    ALT1 = 0b101
    ALT2 = 0b110
    ALT3 = 0b111


class Pull(IntEnum):
    OFF = 0
    DOWN = 1
    UP = 2
    RESERVED = 3


#: BCM2711 reordered the pull encoding relative to the legacy GPPUD register.
_PUP_PDN_TO_PULL = {0b00: Pull.OFF, 0b01: Pull.UP, 0b10: Pull.DOWN, 0b11: Pull.RESERVED}
_PULL_TO_PUP_PDN = {Pull.OFF: 0b00, Pull.UP: 0b01, Pull.DOWN: 0b10, Pull.RESERVED: 0b11}


class BcmGpio(MMIODevice):
    """
    :param pull_style: ``'legacy'`` for BCM2835/2837 (GPPUD + GPPUDCLK
        clocked sequence) or ``'bcm2711'`` for the Pi 4's direct
        GPIO_PUP_PDN_CNTRL registers.
    """

    SIZE = 0x1000

    REGISTERS = {
        0x00: 'GPFSEL0', 0x04: 'GPFSEL1', 0x08: 'GPFSEL2',
        0x0C: 'GPFSEL3', 0x10: 'GPFSEL4', 0x14: 'GPFSEL5',
        0x1C: 'GPSET0', 0x20: 'GPSET1',
        0x28: 'GPCLR0', 0x2C: 'GPCLR1',
        0x34: 'GPLEV0', 0x38: 'GPLEV1',
        0x40: 'GPEDS0', 0x44: 'GPEDS1',
        0x4C: 'GPREN0', 0x50: 'GPREN1',
        0x58: 'GPFEN0', 0x5C: 'GPFEN1',
        0x64: 'GPHEN0', 0x68: 'GPHEN1',
        0x70: 'GPLEN0', 0x74: 'GPLEN1',
        0x7C: 'GPAREN0', 0x80: 'GPAREN1',
        0x88: 'GPAFEN0', 0x8C: 'GPAFEN1',
        0x94: 'GPPUD', 0x98: 'GPPUDCLK0', 0x9C: 'GPPUDCLK1',
        0xE4: 'GPIO_PUP_PDN_CNTRL_REG0', 0xE8: 'GPIO_PUP_PDN_CNTRL_REG1',
        0xEC: 'GPIO_PUP_PDN_CNTRL_REG2', 0xF0: 'GPIO_PUP_PDN_CNTRL_REG3',
    }

    def __init__(self, pull_style='legacy', name='gpio', trace=False):
        super().__init__(self.SIZE, name=name, trace=trace)
        if pull_style not in ('legacy', 'bcm2711'):
            raise ValueError(f'unknown pull_style {pull_style!r}')
        self.pull_style = pull_style

        self._fsel = [GpioFunction.INPUT] * NUM_PINS
        self._output = [False] * NUM_PINS      # value driven by the SoC
        self._external = [None] * NUM_PINS     # value driven by the outside world
        self._pull = [Pull.OFF] * NUM_PINS

        # Event detection configuration and latched status.
        self._rising_en = [False] * NUM_PINS
        self._falling_en = [False] * NUM_PINS
        self._high_en = [False] * NUM_PINS
        self._low_en = [False] * NUM_PINS
        self._async_rising_en = [False] * NUM_PINS
        self._async_falling_en = [False] * NUM_PINS
        self._eds = [False] * NUM_PINS

        # Legacy pull sequence staging.
        self._gppud = Pull.OFF

        #: pin -> list of (sequence_number, level) for output transitions
        self.waveforms = {}
        self._seq = 0
        #: callbacks invoked as fn(pin, level) whenever an output pin changes
        self.output_callbacks = []

        self._last_level = [self.level(p) for p in range(NUM_PINS)]

    # ------------------------------------------------------------------
    # Pin-level API (for tests and virtual hardware)
    # ------------------------------------------------------------------
    def function(self, pin: int) -> GpioFunction:
        """Current FSEL function of ``pin``."""
        return self._fsel[pin]

    def level(self, pin: int) -> bool:
        """
        Effective electrical level of ``pin``.

        An output pin reflects what the SoC drives.  An input pin reflects
        the external driver if one is attached, otherwise its pull resistor.
        """
        if self._fsel[pin] == GpioFunction.OUTPUT:
            return self._output[pin]
        if self._external[pin] is not None:
            return self._external[pin]
        return self._pull[pin] == Pull.UP

    def pull(self, pin: int) -> Pull:
        """Configured pull resistor for ``pin``."""
        return self._pull[pin]

    def drive_input(self, pin: int, level) -> None:
        """
        Drive ``pin`` from the outside world.  Pass ``None`` to release it
        back to its pull resistor.  Triggers edge detection.
        """
        self._external[pin] = None if level is None else bool(level)
        self._refresh_events()

    def transitions(self, pin: int):
        """Recorded ``(seq, level)`` output transitions for ``pin``."""
        return list(self.waveforms.get(pin, []))

    def pulse_count(self, pin: int) -> int:
        """Number of rising edges recorded on ``pin`` -- handy for PWM/bit-bang tests."""
        return sum(1 for _, level in self.waveforms.get(pin, []) if level)

    # ------------------------------------------------------------------
    # Internal state updates
    # ------------------------------------------------------------------
    def _note_transition(self, pin, level):
        self._seq += 1
        self.waveforms.setdefault(pin, []).append((self._seq, level))
        for cb in self.output_callbacks:
            cb(pin, level)

    def _refresh_events(self):
        """Recompute latched edge/level events and the IRQ line."""
        for pin in range(NUM_PINS):
            now = self.level(pin)
            before = self._last_level[pin]
            if now != before:
                if now and (self._rising_en[pin] or self._async_rising_en[pin]):
                    self._eds[pin] = True
                if not now and (self._falling_en[pin] or self._async_falling_en[pin]):
                    self._eds[pin] = True
                if self._fsel[pin] == GpioFunction.OUTPUT:
                    self._note_transition(pin, now)
            if now and self._high_en[pin]:
                self._eds[pin] = True
            if not now and self._low_en[pin]:
                self._eds[pin] = True
            self._last_level[pin] = now
        self.set_irq(any(self._eds))

    def _apply_pull(self, pin, pull):
        self._pull[pin] = pull
        self._refresh_events()

    # ------------------------------------------------------------------
    # Bitmap helpers -- registers come in pairs covering pins 0-31, 32-53
    # ------------------------------------------------------------------
    @staticmethod
    def _bank_range(bank):
        start = 32 * bank
        return range(start, min(start + 32, NUM_PINS))

    def _pack(self, flags, bank):
        value = 0
        for pin in self._bank_range(bank):
            if flags[pin]:
                value |= 1 << (pin - 32 * bank)
        return value

    def _unpack_into(self, flags, bank, value):
        for pin in self._bank_range(bank):
            flags[pin] = bool(value & (1 << (pin - 32 * bank)))

    # ------------------------------------------------------------------
    # Register interface
    # ------------------------------------------------------------------
    def read_register(self, offset):
        if 0x00 <= offset <= 0x14:                       # GPFSELn
            base = (offset // 4) * 10
            value = 0
            for i in range(10):
                pin = base + i
                if pin < NUM_PINS:
                    value |= int(self._fsel[pin]) << (3 * i)
            return value
        if offset in (0x34, 0x38):                       # GPLEVn
            bank = (offset - 0x34) // 4
            return self._pack([self.level(p) for p in range(NUM_PINS)], bank)
        if offset in (0x40, 0x44):                       # GPEDSn
            return self._pack(self._eds, (offset - 0x40) // 4)
        if offset in (0x4C, 0x50):
            return self._pack(self._rising_en, (offset - 0x4C) // 4)
        if offset in (0x58, 0x5C):
            return self._pack(self._falling_en, (offset - 0x58) // 4)
        if offset in (0x64, 0x68):
            return self._pack(self._high_en, (offset - 0x64) // 4)
        if offset in (0x70, 0x74):
            return self._pack(self._low_en, (offset - 0x70) // 4)
        if offset in (0x7C, 0x80):
            return self._pack(self._async_rising_en, (offset - 0x7C) // 4)
        if offset in (0x88, 0x8C):
            return self._pack(self._async_falling_en, (offset - 0x88) // 4)
        if offset == 0x94 and self.pull_style == 'legacy':
            return int(self._gppud)
        if 0xE4 <= offset <= 0xF0 and self.pull_style == 'bcm2711':
            base = ((offset - 0xE4) // 4) * 16
            value = 0
            for i in range(16):
                pin = base + i
                if pin < NUM_PINS:
                    value |= _PULL_TO_PUP_PDN[self._pull[pin]] << (2 * i)
            return value
        # GPSET/GPCLR/GPPUDCLK read as zero on real silicon.
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        if 0x00 <= offset <= 0x14:                       # GPFSELn
            base = (offset // 4) * 10
            for i in range(10):
                pin = base + i
                if pin < NUM_PINS:
                    self._fsel[pin] = GpioFunction((value >> (3 * i)) & 0b111)
            self._refresh_events()
            return
        if offset in (0x1C, 0x20):                       # GPSETn
            bank = (offset - 0x1C) // 4
            for pin in self._bank_range(bank):
                if value & (1 << (pin - 32 * bank)):
                    self._output[pin] = True
            self._refresh_events()
            return
        if offset in (0x28, 0x2C):                       # GPCLRn
            bank = (offset - 0x28) // 4
            for pin in self._bank_range(bank):
                if value & (1 << (pin - 32 * bank)):
                    self._output[pin] = False
            self._refresh_events()
            return
        if offset in (0x40, 0x44):                       # GPEDSn -- write 1 to clear
            bank = (offset - 0x40) // 4
            for pin in self._bank_range(bank):
                if value & (1 << (pin - 32 * bank)):
                    self._eds[pin] = False
            self.set_irq(any(self._eds))
            return
        if offset in (0x4C, 0x50):
            self._unpack_into(self._rising_en, (offset - 0x4C) // 4, value)
            return
        if offset in (0x58, 0x5C):
            self._unpack_into(self._falling_en, (offset - 0x58) // 4, value)
            return
        if offset in (0x64, 0x68):
            self._unpack_into(self._high_en, (offset - 0x64) // 4, value)
            self._refresh_events()
            return
        if offset in (0x70, 0x74):
            self._unpack_into(self._low_en, (offset - 0x70) // 4, value)
            self._refresh_events()
            return
        if offset in (0x7C, 0x80):
            self._unpack_into(self._async_rising_en, (offset - 0x7C) // 4, value)
            return
        if offset in (0x88, 0x8C):
            self._unpack_into(self._async_falling_en, (offset - 0x88) // 4, value)
            return
        if self.pull_style == 'legacy':
            if offset == 0x94:                           # GPPUD
                self._gppud = Pull(value & 0b11)
                return
            if offset in (0x98, 0x9C):                   # GPPUDCLKn
                bank = (offset - 0x98) // 4
                for pin in self._bank_range(bank):
                    if value & (1 << (pin - 32 * bank)):
                        self._apply_pull(pin, self._gppud)
                return
        elif 0xE4 <= offset <= 0xF0:                     # BCM2711 direct control
            base = ((offset - 0xE4) // 4) * 16
            for i in range(16):
                pin = base + i
                if pin < NUM_PINS:
                    self._apply_pull(pin, _PUP_PDN_TO_PULL[(value >> (2 * i)) & 0b11])
            self._refresh_events()
            return
