"""
Tegra X1 (Jetson Nano) GPIO controller.

The Tegra GPIO block is organised very differently from Broadcom's.  There
are 8 controllers, each covering 4 ports of 8 pins (256 pins total).  Within
a controller, each register is replicated once per port at 4-byte intervals,
and controllers are 0x100 apart::

    controller_base = 0x6000D000 + controller * 0x100
    register_addr   = controller_base + reg_offset + port * 4

Each pin also has a *mode* bit (CNF): 1 selects GPIO, 0 hands the pin to the
SFIO peripheral muxing -- the rough equivalent of Broadcom's ALT functions.

The ``MSK_*`` aliases at +0x80 are the important quirk: writing them uses
the upper 8 bits as a write-enable mask for the lower 8 bits, so firmware
can update one pin without a read-modify-write.  Linux's ``gpio-tegra``
driver uses these almost exclusively, so any realistic test needs them.
"""

from armulator.peripherals.mmio import MMIODevice

NUM_CONTROLLERS = 8
PORTS_PER_CONTROLLER = 4
PINS_PER_PORT = 8
NUM_PINS = NUM_CONTROLLERS * PORTS_PER_CONTROLLER * PINS_PER_PORT  # 256

# Register offsets within a controller (each replicated per port at +0,4,8,C)
CNF = 0x00
OE = 0x10
OUT = 0x20
IN = 0x30
INT_STA = 0x40
INT_ENB = 0x50
INT_LVL = 0x60
INT_CLR = 0x70
MSK_CNF = 0x80
MSK_OE = 0x90
MSK_OUT = 0xA0
MSK_INT_STA = 0xC0
MSK_INT_ENB = 0xD0
MSK_INT_LVL = 0xE0

#: Masked register -> the plain register it aliases.
_MASKED_ALIASES = {
    MSK_CNF: CNF, MSK_OE: OE, MSK_OUT: OUT,
    MSK_INT_STA: INT_STA, MSK_INT_ENB: INT_ENB, MSK_INT_LVL: INT_LVL,
}

_REG_NAMES = {
    CNF: 'CNF', OE: 'OE', OUT: 'OUT', IN: 'IN',
    INT_STA: 'INT_STA', INT_ENB: 'INT_ENB', INT_LVL: 'INT_LVL', INT_CLR: 'INT_CLR',
    MSK_CNF: 'MSK_CNF', MSK_OE: 'MSK_OE', MSK_OUT: 'MSK_OUT',
    MSK_INT_STA: 'MSK_INT_STA', MSK_INT_ENB: 'MSK_INT_ENB', MSK_INT_LVL: 'MSK_INT_LVL',
}


class TegraGpio(MMIODevice):
    """
    Tegra X1 GPIO with the same pin-level test API as :class:`BcmGpio`.

    Pins are addressed by flat index 0-255.  Use :meth:`pin_number` to
    convert from the datasheet's port-letter notation (``PA0``, ``PBB4``).
    """

    SIZE = NUM_CONTROLLERS * 0x100

    def __init__(self, name='tegra_gpio', trace=False):
        super().__init__(self.SIZE, name=name, trace=trace)
        self._gpio_mode = [False] * NUM_PINS   # CNF: True = GPIO, False = SFIO
        self._output_en = [False] * NUM_PINS   # OE
        self._output = [False] * NUM_PINS      # OUT
        self._external = [None] * NUM_PINS
        self._int_en = [False] * NUM_PINS
        self._int_lvl = [0] * NUM_PINS
        self._int_sta = [False] * NUM_PINS

        self.waveforms = {}
        self._seq = 0
        self.output_callbacks = []
        self._last_level = [False] * NUM_PINS

    # ------------------------------------------------------------------
    # Pin naming
    # ------------------------------------------------------------------
    @staticmethod
    def pin_number(name: str) -> int:
        """
        Convert Tegra pin notation to a flat index.

        ``'PA0'`` -> 0, ``'PB0'`` -> 8, ``'PZ7'`` -> 207,
        ``'PAA0'`` -> 208, ``'PBB0'`` -> 216.
        """
        name = name.upper().lstrip('P')
        letters, digits = '', ''
        for ch in name:
            if ch.isdigit():
                digits += ch
            else:
                letters += ch
        if not letters or not digits:
            raise ValueError(f'bad Tegra pin name {name!r}')
        if len(letters) == 1:
            port = ord(letters) - ord('A')
        elif len(letters) == 2 and letters[0] == letters[1]:
            port = 26 + (ord(letters[0]) - ord('A'))
        else:
            raise ValueError(f'bad Tegra port {letters!r}')
        offset = int(digits)
        if not 0 <= offset < PINS_PER_PORT:
            raise ValueError(f'pin offset {offset} out of range')
        return port * PINS_PER_PORT + offset

    # ------------------------------------------------------------------
    # Pin-level API
    # ------------------------------------------------------------------
    def is_gpio(self, pin) -> bool:
        """True if the pin is in GPIO mode rather than handed to SFIO mux."""
        return self._gpio_mode[self._resolve(pin)]

    def is_output(self, pin) -> bool:
        pin = self._resolve(pin)
        return self._gpio_mode[pin] and self._output_en[pin]

    def level(self, pin) -> bool:
        pin = self._resolve(pin)
        if self.is_output(pin):
            return self._output[pin]
        if self._external[pin] is not None:
            return self._external[pin]
        return False

    def drive_input(self, pin, level) -> None:
        """Drive a pin from the outside world (``None`` releases it)."""
        pin = self._resolve(pin)
        self._external[pin] = None if level is None else bool(level)
        self._refresh()

    def transitions(self, pin):
        return list(self.waveforms.get(self._resolve(pin), []))

    @staticmethod
    def _resolve(pin):
        return TegraGpio.pin_number(pin) if isinstance(pin, str) else pin

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _refresh(self):
        for pin in range(NUM_PINS):
            now = self.level(pin)
            before = self._last_level[pin]
            if now != before:
                if self.is_output(pin):
                    self._seq += 1
                    self.waveforms.setdefault(pin, []).append((self._seq, now))
                    for cb in self.output_callbacks:
                        cb(pin, now)
                elif self._int_en[pin]:
                    # INT_LVL: bit 0 selects high/rising, edge bits at +8/+16
                    self._int_sta[pin] = True
            self._last_level[pin] = now
        self.set_irq(any(self._int_sta))

    def _decode(self, offset):
        """Return ``(reg, base_pin)`` for a register offset, or ``None``."""
        controller, within = divmod(offset, 0x100)
        if controller >= NUM_CONTROLLERS:
            return None
        reg = within & ~0xF
        port = (within & 0xF) // 4
        base_pin = (controller * PORTS_PER_CONTROLLER + port) * PINS_PER_PORT
        return reg, base_pin

    def register_name(self, offset):
        decoded = self._decode(offset)
        if decoded is None:
            return f'+0x{offset:03X}'
        reg, base_pin = self._decode(offset)
        port = base_pin // PINS_PER_PORT
        letter = chr(ord('A') + port) if port < 26 else chr(ord('A') + port - 26) * 2
        return f'{_REG_NAMES.get(reg, hex(reg))}_P{letter}'

    def _pack(self, flags, base_pin):
        return sum(1 << i for i in range(PINS_PER_PORT) if flags[base_pin + i])

    def _unpack(self, flags, base_pin, value):
        for i in range(PINS_PER_PORT):
            flags[base_pin + i] = bool(value & (1 << i))

    # ------------------------------------------------------------------
    # Register interface
    # ------------------------------------------------------------------
    def read_register(self, offset):
        decoded = self._decode(offset)
        if decoded is None:
            return self.DEFAULT_READ
        reg, base = decoded
        reg = _MASKED_ALIASES.get(reg, reg)   # masked regs read like their alias
        if reg == CNF:
            return self._pack(self._gpio_mode, base)
        if reg == OE:
            return self._pack(self._output_en, base)
        if reg == OUT:
            return self._pack(self._output, base)
        if reg == IN:
            return sum(1 << i for i in range(PINS_PER_PORT) if self.level(base + i))
        if reg == INT_STA:
            return self._pack(self._int_sta, base)
        if reg == INT_ENB:
            return self._pack(self._int_en, base)
        if reg == INT_LVL:
            return sum(self._int_lvl[base + i] << i for i in range(PINS_PER_PORT))
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        decoded = self._decode(offset)
        if decoded is None:
            return
        reg, base = decoded

        if reg in _MASKED_ALIASES:
            # Upper 8 bits are the write-enable mask for the lower 8.
            mask = (value >> 8) & 0xFF
            data = value & 0xFF
            target = _MASKED_ALIASES[reg]
            current = self.read_register((offset & ~0xFF) | target | (offset & 0xC))
            value = (current & ~mask) | (data & mask)
            reg = target

        if reg == CNF:
            self._unpack(self._gpio_mode, base, value)
        elif reg == OE:
            self._unpack(self._output_en, base, value)
        elif reg == OUT:
            self._unpack(self._output, base, value)
        elif reg == INT_ENB:
            self._unpack(self._int_en, base, value)
        elif reg == INT_LVL:
            for i in range(PINS_PER_PORT):
                self._int_lvl[base + i] = (value >> i) & 1
        elif reg == INT_CLR:
            for i in range(PINS_PER_PORT):
                if value & (1 << i):
                    self._int_sta[base + i] = False
            self.set_irq(any(self._int_sta))
            return
        self._refresh()
