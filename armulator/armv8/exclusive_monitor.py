"""
The exclusive monitor.

``LDXR`` takes a reservation on a block of memory and ``STXR`` succeeds only if that
reservation is still intact. On a single core this can be faked - nothing else can
interfere, so the store always succeeds - but with several cores it becomes the thing
that makes a spinlock a lock rather than a suggestion.

Two rules do the work:

* A successful ``STXR`` clears every core's reservation covering that block, so only one
  of several racing cores can win.
* An ordinary store also clears other cores' reservations. Without this, a core could
  release a lock with a plain ``STR`` while another core held a stale reservation and
  then "succeeded" at acquiring a lock that had moved on underneath it.

Reservations are tracked per block rather than per byte. The exclusive reservation
granule on a Cortex-A57 is 16 bytes, which means two variables in the same 16 bytes can
make each other's exclusive stores fail - real, occasionally surprising, and worth
reproducing rather than smoothing over.
"""

#: Exclusive reservation granule, in bytes, as implemented by the Cortex-A57.
RESERVATION_GRANULE = 16


class ExclusiveMonitor:
    """
    Reservation state shared by every core in a cluster.
    """

    def __init__(self, granule=RESERVATION_GRANULE):
        self.granule = granule
        #: cpu id -> the reserved block address, or None when no reservation is held.
        self.reservations = {}
        #: Counters, useful for showing that contention actually happened.
        self.successes = 0
        self.failures = 0

    def _block(self, address):
        return address & ~(self.granule - 1)

    def reserve(self, cpu_id, address):
        """Take a reservation for ``cpu_id`` (LDXR)."""
        self.reservations[cpu_id] = self._block(address)

    def clear(self, cpu_id):
        """Drop this core's reservation (CLREX, or taking an exception)."""
        self.reservations.pop(cpu_id, None)

    def check_and_clear(self, cpu_id, address):
        """
        Attempt an exclusive store (STXR). Returns True when it succeeds.

        On success every core's reservation for the block is cleared, so a second core
        racing for the same lock will fail its own store and retry.
        """
        block = self._block(address)
        held = self.reservations.get(cpu_id)
        if held != block:
            self.failures += 1
            # A failed store still loses this core's reservation.
            self.reservations.pop(cpu_id, None)
            return False

        for other, reserved in list(self.reservations.items()):
            if reserved == block:
                del self.reservations[other]
        self.successes += 1
        return True

    def notify_store(self, cpu_id, address, size):
        """
        An ordinary store happened. Clear any *other* core's reservation that overlaps
        it - the storing core keeps its own, since the architecture only requires
        remote observers to lose theirs.
        """
        if not self.reservations:
            return
        first = self._block(address)
        last = self._block(address + size - 1)
        for other, reserved in list(self.reservations.items()):
            if other != cpu_id and first <= reserved <= last:
                del self.reservations[other]

    def __repr__(self):
        held = ', '.join(f'cpu{cpu}=0x{block:X}' for cpu, block in sorted(self.reservations.items()))
        return f'<ExclusiveMonitor {held or "no reservations"}>'
