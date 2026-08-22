"""
The AArch64 architected generic timer (EL1 physical timer).

Every ARMv8-A core has one, and it is how bare-metal code gets a periodic tick
without touching an SoC peripheral -- which is the point: the same driver runs
on a Jetson, a Pi and a virt machine, because the timer is part of the
architecture rather than part of the board.

Three registers matter:

``CNTPCT_EL0``
    A free-running counter, incrementing at ``CNTFRQ_EL0`` Hz.
``CNTP_CVAL_EL0``
    A compare value. ``CNTP_TVAL_EL0`` is the same thing written as a delta:
    writing it sets ``CVAL = CNTPCT + TVAL``, and reading it returns
    ``CVAL - CNTPCT``, which is how a driver rearms the timer for "another N
    ticks from now".
``CNTP_CTL_EL0``
    ``ENABLE`` (bit 0), ``IMASK`` (bit 1) and the read-only ``ISTATUS``
    (bit 2). ISTATUS is *not* stored -- it is recomputed from the counter, so
    it becomes true the moment ``CNTPCT >= CVAL`` whether or not anyone is
    looking. The interrupt line asserts when the condition holds, the timer is
    enabled, and IMASK is clear.

The counter here is virtual: it advances by :meth:`tick` as instructions
retire rather than by wall clock, so tests are deterministic and a firmware
delay loop terminates in a bounded number of steps instead of depending on how
fast the host happens to be.

The output is a PPI -- interrupt ID 30 for the EL1 physical timer -- which is
private to each core rather than routed by the distributor's target registers.
"""

#: Interrupt ID of the EL1 physical timer. A PPI, so it is per-core.
TIMER_PPI = 30

CTL_ENABLE = 1 << 0
CTL_IMASK = 1 << 1
CTL_ISTATUS = 1 << 2


class GenericTimer:
    """
    One core's EL1 physical timer.

    :param frequency: ``CNTFRQ_EL0``, in Hz.
    :param ticks_per_instruction: how far :meth:`tick` advances the counter
        for each retired instruction. The architected counter runs far slower
        than the core, but a 1:1 ratio would make a firmware busy-wait for a
        millisecond take millions of emulated instructions, so the default
        trades fidelity for tests that finish.
    """

    def __init__(self, frequency=19200000, ticks_per_instruction=64):
        self.frequency = frequency
        self.ticks_per_instruction = ticks_per_instruction
        #: CNTPCT_EL0.
        self.count = 0
        #: CNTP_CVAL_EL0.
        self.compare = 0
        self._enabled = False
        self._masked = False

    # ------------------------------------------------------------------
    # Counter
    # ------------------------------------------------------------------
    def tick(self, amount=None) -> None:
        """Advance the counter, by one instruction's worth if unspecified."""
        if amount is None:
            amount = self.ticks_per_instruction
        self.count = (self.count + amount) & 0xFFFFFFFFFFFFFFFF

    # ------------------------------------------------------------------
    # Condition and output
    # ------------------------------------------------------------------
    @property
    def istatus(self) -> bool:
        """True when the compare condition holds, regardless of IMASK."""
        return self._enabled and self.count >= self.compare

    @property
    def irq_pending(self) -> bool:
        """The level of the timer's interrupt output."""
        return self.istatus and not self._masked

    # ------------------------------------------------------------------
    # Register faces
    # ------------------------------------------------------------------
    @property
    def ctl(self) -> int:
        value = 0
        if self._enabled:
            value |= CTL_ENABLE
        if self._masked:
            value |= CTL_IMASK
        if self.istatus:
            value |= CTL_ISTATUS
        return value

    @ctl.setter
    def ctl(self, value: int) -> None:
        # ISTATUS is read-only; it is recomputed, never stored.
        self._enabled = bool(value & CTL_ENABLE)
        self._masked = bool(value & CTL_IMASK)

    @property
    def tval(self) -> int:
        """``CVAL - CNTPCT``, as a signed 32-bit value."""
        delta = (self.compare - self.count) & 0xFFFFFFFF
        return delta

    @tval.setter
    def tval(self, value: int) -> None:
        # A 32-bit signed delta from now.
        delta = value & 0xFFFFFFFF
        if delta & 0x80000000:
            delta -= 1 << 32
        self.compare = (self.count + delta) & 0xFFFFFFFFFFFFFFFF
