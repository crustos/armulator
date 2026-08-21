"""
Memory-mapped I/O device framework for armulator.

``MemoryType`` (armulator.armv6.memory_types) is the extension point the
memory controller hub dispatches through: the hub locates a controller by
physical address, then calls ``read(offset, size)`` / ``write(offset, size,
value)`` with *byte strings*.  ``MMIODevice`` sits on top of that and gives
peripheral models a far more convenient contract:

    * accesses are decoded to naturally-aligned 32-bit register offsets
    * subclasses implement ``read_register`` / ``write_register`` in ints
    * every access is optionally recorded for test assertions
    * devices expose an IRQ line that a board can poll

Sub-word accesses are handled by read-modify-write so that byte writes to
things like the PL011 data register behave the way firmware expects.
"""

from abc import abstractmethod
from collections import namedtuple

from armulator.armv6.memory_types import MemoryType

#: One recorded register access. ``kind`` is 'r' or 'w'.
Access = namedtuple('Access', ['kind', 'offset', 'size', 'value', 'name'])


class MMIODevice(MemoryType):
    """
    Base class for memory-mapped peripherals.

    Subclasses should define ``REGISTERS`` (offset -> name) for readable
    traces, and implement :meth:`read_register` and :meth:`write_register`.
    """

    #: Mapping of register offset to human readable name, for tracing.
    REGISTERS: dict = {}

    #: Value returned for offsets the device does not implement.
    DEFAULT_READ = 0x00000000

    def __init__(self, size, name=None, trace=False):
        super().__init__(size)
        self.name = name or type(self).__name__
        self.trace = trace
        #: Chronological list of :class:`Access` records (when ``trace``).
        self.accesses = []
        #: Level of this device's interrupt output line.
        self.irq_pending = False

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------
    @abstractmethod
    def read_register(self, offset: int) -> int:
        """Return the 32-bit value of the register at ``offset``."""

    @abstractmethod
    def write_register(self, offset: int, value: int) -> None:
        """Write the 32-bit ``value`` to the register at ``offset``."""

    def register_name(self, offset: int) -> str:
        return self.REGISTERS.get(offset, f'+0x{offset:03X}')

    # ------------------------------------------------------------------
    # Interrupt line
    # ------------------------------------------------------------------
    def set_irq(self, level: bool) -> None:
        self.irq_pending = bool(level)

    # ------------------------------------------------------------------
    # MemoryType plumbing
    # ------------------------------------------------------------------
    def _record(self, kind, offset, size, value):
        if self.trace:
            self.accesses.append(
                Access(kind, offset, size, value, self.register_name(offset & ~0x3))
            )

    def read(self, address, size):
        """Return ``size`` bytes at ``address`` (little endian)."""
        value = 0
        for i in range(0, size, 4):
            word_offset = (address + i) & ~0x3
            word = self.read_register(word_offset) & 0xFFFFFFFF
            value |= word << (8 * i)
        # Shift down for accesses that start mid-word (e.g. byte reads).
        value >>= 8 * (address & 0x3)
        value &= (1 << (8 * size)) - 1
        self._record('r', address, size, value)
        return value.to_bytes(size, 'little')

    def write(self, address, size, value):
        """Write ``size`` bytes of ``value`` at ``address`` (little endian)."""
        if isinstance(value, (bytes, bytearray)):
            value = int.from_bytes(value, 'little')
        self._record('w', address, size, value)

        if size >= 4 and (address & 0x3) == 0:
            for i in range(0, size, 4):
                word = (value >> (8 * i)) & 0xFFFFFFFF
                self.write_register(address + i, word)
            return

        # Sub-word or unaligned: read-modify-write the containing word.
        word_offset = address & ~0x3
        byte_in_word = address & 0x3
        current = self.read_register(word_offset) & 0xFFFFFFFF
        mask = ((1 << (8 * size)) - 1) << (8 * byte_in_word)
        mask &= 0xFFFFFFFF
        merged = (current & ~mask) | ((value << (8 * byte_in_word)) & mask)
        self.write_register(word_offset, merged & 0xFFFFFFFF)

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------
    def clear_trace(self) -> None:
        self.accesses.clear()

    def writes_to(self, name: str):
        """All traced writes whose register name matches ``name``."""
        return [a for a in self.accesses if a.kind == 'w' and a.name == name]

    def reads_of(self, name: str):
        """All traced reads whose register name matches ``name``."""
        return [a for a in self.accesses if a.kind == 'r' and a.name == name]

    def format_trace(self) -> str:
        lines = []
        for a in self.accesses:
            arrow = '<-' if a.kind == 'w' else '->'
            lines.append(f'{self.name}.{a.name} {arrow} 0x{a.value:08X}')
        return '\n'.join(lines)


class UnimplementedDevice(MMIODevice):
    """
    A stub that reads as zero and swallows writes.

    Useful for filling in peripherals the firmware under test touches
    incidentally, so it does not fall off the end of the memory map.
    """

    def read_register(self, offset):
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        pass
