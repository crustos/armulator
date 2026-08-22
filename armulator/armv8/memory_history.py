"""
A bounded history of what memory used to contain.

Modelling a reordered load means answering it with the value memory held at some earlier
moment, which requires remembering what that was. This keeps a short per-byte log of
writes; anything older than the horizon is forgotten, since no load is ever allowed to
look back that far.

History is recorded at the point a write becomes globally visible - when it leaves a
store buffer, not when the instruction issued - because that is the moment another core
could first have seen it.
"""

from bisect import bisect_right

#: How far back a load may be answered from, in global instruction ticks. Beyond this
#: the log is pruned; a lookup older than the horizon falls back to current memory,
#: which is safe because it can only ever make a reordering less visible, not more.
DEFAULT_HORIZON = 512


class Clock:
    """
    A tick shared by every core in a cluster.

    Per-core instruction counters cannot order events across cores - two cores each on
    their hundredth instruction have not necessarily reached the same moment. One shared
    counter gives every write and every read a comparable timestamp.
    """

    def __init__(self):
        self.now = 0

    def tick(self):
        self.now += 1
        return self.now


class MemoryHistory:
    """
    Per-byte write log, newest last.
    """

    def __init__(self, horizon=DEFAULT_HORIZON):
        self.horizon = horizon
        #: byte address -> (list of times, list of values), kept sorted by time.
        self._times = {}
        self._values = {}
        self._last_prune = 0

    #: Timestamp used for the value a byte held before anything was ever recorded for
    #: it. Earlier than any real tick, so a lookup at time zero still finds it.
    BASELINE = -1

    def record(self, address, size, value, now, previous=None):
        """
        Note that ``value`` became visible at ``address`` at time ``now``.

        ``previous`` is what those bytes held immediately beforehand. It seeds the log
        the first time a byte is written, and without it a load looking back to before
        that first write has nothing to find and falls through to current memory -
        which quietly cancels the reordering it was asked to model.
        """
        for offset in range(size):
            byte_address = address + offset
            byte = (value >> (8 * offset)) & 0xFF
            times = self._times.setdefault(byte_address, [])
            values = self._values.setdefault(byte_address, [])
            if not times and previous is not None:
                times.append(self.BASELINE)
                values.append((previous >> (8 * offset)) & 0xFF)
            if times and times[-1] == now:
                values[-1] = byte
            else:
                times.append(now)
                values.append(byte)
        if now - self._last_prune > self.horizon:
            self.prune(now)

    def byte_as_of(self, byte_address, when):
        """
        The value of one byte at time ``when``, or None when nothing is recorded that
        far back and the caller should use current memory.
        """
        times = self._times.get(byte_address)
        if not times:
            return None
        index = bisect_right(times, when)
        if index == 0:
            # Every recorded write is newer than the time asked about, so the value then
            # was whatever preceded them - which is no longer in the log.
            return None
        return self._values[byte_address][index - 1]

    def value_as_of(self, address, size, when, current):
        """
        Reconstruct a ``size``-byte value as of ``when``, filling gaps from ``current``.
        """
        result = current
        changed = False
        for offset in range(size):
            byte = self.byte_as_of(address + offset, when)
            if byte is None:
                continue
            shift = 8 * offset
            result = (result & ~(0xFF << shift)) | (byte << shift)
            changed = True
        return result if changed else current

    def prune(self, now):
        """Forget everything older than the horizon, keeping one entry per byte."""
        cutoff = now - self.horizon
        self._last_prune = now
        for byte_address, times in list(self._times.items()):
            index = bisect_right(times, cutoff)
            # Keep the last entry at or before the cutoff so a lookup on the boundary
            # still has something to return.
            keep_from = max(index - 1, 0)
            if keep_from:
                self._times[byte_address] = times[keep_from:]
                self._values[byte_address] = self._values[byte_address][keep_from:]

    def clear(self):
        self._times.clear()
        self._values.clear()
        self._last_prune = 0
