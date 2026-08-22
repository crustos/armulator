"""
Store buffering, so that a missing barrier has consequences.

Execution here is strictly in order, so without help every core sees every other core's
writes the instant they happen. That is sequential consistency, which is *stronger* than
anything real hardware provides - and it means lock-free code with missing barriers runs
perfectly in the emulator and then fails on silicon. The whole class of bug the barriers
exist to prevent is invisible.

A per-core store buffer fixes that. A write lands in the issuing core's buffer, where
only that core can see it, and reaches memory later. Until it drains, other cores read
the old value. That single mechanism is enough to reproduce the classic failures:

* **Store then load reordering.** A core buffers its write and then reads someone else's
  location, which has not been updated either. Both cores read zero - the outcome
  sequential consistency forbids and every real weak-memory machine allows. This is what
  breaks Dekker-style mutual exclusion written without barriers.

* **Store then store reordering.** Two buffered writes reach memory in an order other
  than the one the program issued, so a reader sees a flag set before the data it was
  supposed to be guarding.

What this is *not* is a model of the Cortex-A57's actual memory model. Real reordering is
non-deterministic and depends on cache state, timing and speculation. Reproducing that
faithfully would make failures unrepeatable, which is the opposite of useful in a test.
So the policies below are deterministic and, in the strongest setting, deliberately
adversarial: they reorder whenever the architecture permits it rather than whenever a
real chip happens to. Code that survives that is code that does not depend on ordering it
never established.
"""

from enum import Enum


class MemoryModel(Enum):
    """
    How aggressively stores are allowed to be reordered.
    """

    #: Every store goes straight to memory. Sequentially consistent, and stronger than
    #: any real machine - the default, so existing behaviour is unchanged.
    SEQUENTIAL = 'sequential'

    #: Stores are buffered and drained in program order. Exposes store/load reordering,
    #: which is the failure behind Dekker and Peterson written without barriers.
    RELAXED = 'relaxed'

    #: Stores are buffered and drained in reverse program order. Also exposes
    #: store/store reordering, so a flag published before its data is caught. Reorders
    #: at every opportunity the architecture permits rather than imitating a real chip.
    ADVERSARIAL = 'adversarial'


#: How many instructions a store sits in the buffer before it may drain on its own.
#:
#: Some delay is essential. With none, a store would reach memory before the very next
#: instruction and no reordering would ever be observable. With an unbounded delay, a
#: program ending in a plain store would never publish it and a reader would spin
#: forever - which looks like a memory-ordering bug but is really just a stalled model.
#: A small bounded window gives both: reordering is visible, and every store lands.
#:
#: The value is a detection window, not a hardware figure. Too small and a reordering
#: lasts fewer instructions than a polling loop takes to come round, so a real bug slips
#: past unobserved; too large and programs spend their time waiting on the buffer. Eight
#: catches the common release-ordering mistakes; widen it to hunt for rarer ones.
DEFAULT_STORE_LATENCY = 8


class StoreBuffer:
    """
    One core's pending stores.

    Entries are byte-granular on lookup so that a load can be satisfied partly from the
    buffer and partly from memory - which is what happens when a byte store is followed
    by a word load covering it.
    """

    def __init__(self, capacity=16, model=MemoryModel.SEQUENTIAL,
                 latency=DEFAULT_STORE_LATENCY):
        self.capacity = capacity
        self.model = model
        self.latency = latency
        #: Pending stores, oldest first, as (issued_at, address, size, value).
        self.entries = []
        #: When a store last retired, so retirements are spaced rather than bunched.
        self.last_retire = 0
        #: Counters for showing that buffering actually happened.
        self.buffered = 0
        self.drained = 0

    def __len__(self):
        return len(self.entries)

    @property
    def buffering(self) -> bool:
        return self.model is not MemoryModel.SEQUENTIAL

    def push(self, address, size, value, now=0):
        """
        Add a store. Returns True when it was buffered, False when the caller should
        write it through to memory itself.
        """
        if not self.buffering:
            return False
        self.entries.append((now, address, size, value))
        self.buffered += 1
        return len(self.entries) <= self.capacity

    def forward(self, address, size):
        """
        Satisfy a load from pending stores where possible.

        Returns a dict of address -> byte for the bytes this buffer can supply. The
        caller fills the rest from memory. A core always sees its own stores, buffered or
        not, so single-threaded code is unaffected by any of this.
        """
        available = {}
        # Later stores win, so walk oldest to newest and let each overwrite.
        for _, entry_address, entry_size, entry_value in self.entries:
            for offset in range(entry_size):
                byte_address = entry_address + offset
                if address <= byte_address < address + size:
                    available[byte_address] = (entry_value >> (8 * offset)) & 0xFF
        return available

    @staticmethod
    def _overlaps(first, second):
        _, first_address, first_size, _ = first
        _, second_address, second_size, _ = second
        return (first_address < second_address + second_size
                and second_address < first_address + first_size)

    def drain_order(self):
        """
        The order pending stores reach memory.

        Under the adversarial policy this is reverse program order - the most demanding
        thing the architecture allows between two plain writes - except that two stores
        to the *same* location keep their program order. Coherence guarantees that writes
        to a single location are observed in a consistent order, so reordering those is
        not weak memory behaviour but a broken machine: a core would see its own writes
        go backwards.
        """
        if self.model is not MemoryModel.ADVERSARIAL:
            return list(self.entries)

        ordered = []
        remaining = list(self.entries)
        while remaining:
            # Take the newest entry that no older entry overlaps, so the reordering is
            # as aggressive as possible while each address stays in program order.
            for index in range(len(remaining) - 1, -1, -1):
                if not any(self._overlaps(remaining[earlier], remaining[index])
                           for earlier in range(index)):
                    ordered.append(remaining.pop(index))
                    break
            else:                                   # pragma: no cover - unreachable
                ordered.append(remaining.pop(0))
        return ordered

    def drain(self, write):
        """
        Flush every pending store through ``write(address, size, value)``.

        Barriers and synchronising instructions use this: it ignores the latency, since
        the point of a barrier is to stop waiting.
        """
        pending = self.drain_order()
        self.entries = []
        for _, address, size, value in pending:
            write(address, size, value)
            self.drained += 1
        return len(pending)

    def retire(self, now, write, limit=1):
        """
        Let up to ``limit`` stores drain of their own accord.

        Called once per instruction, this is what gives a store a bounded lifetime in
        the buffer. Eligibility is judged on the *oldest* pending store rather than on
        each entry individually: once the buffer has been holding anything long enough,
        it releases in its own drain order. Checking each entry's own age instead would
        quietly re-impose program order, because the earliest store is always the first
        to come of age - and then no store/store reordering could ever be observed.
        """
        if not self.entries:
            return 0
        oldest = min(entry[0] for entry in self.entries)
        # Retirements are spaced by the same latency, so two reordered stores land a
        # window apart rather than back to back. Without the spacing the reordering is
        # real but lasts a single instruction, and a reader polling in a short loop
        # steps over it - the bug would be there and still go unobserved.
        since = now - max(oldest, self.last_retire)
        if since < self.latency:
            return 0

        retired = 0
        for entry in self.drain_order():
            if retired >= limit:
                break
            _, address, size, value = entry
            self.entries.remove(entry)
            write(address, size, value)
            self.drained += 1
            retired += 1
        if retired:
            self.last_retire = now
        return retired

    def clear(self):
        self.entries = []
        self.last_retire = 0

    def __repr__(self):
        return f'<StoreBuffer {self.model.value} {len(self.entries)} pending>'


class LoadReorderer:
    """
    Decides how far back in time a load is allowed to look.

    Store buffering cannot produce every weak-memory failure. When a writer orders its
    stores correctly with a barrier, no consistent snapshot of memory shows the flag set
    without the data behind it - and yet real ARM readers still see exactly that, because
    the load of the data is speculated past the loop waiting on the flag. Two loads, and
    the later one effectively happened first.

    Reproducing that needs a rule that distinguishes the two loads, because they need
    opposite treatment: the spin must read fresh values or it never makes progress, while
    the hoisted load must read a stale one or the bug never appears. A uniform lag gives
    one or the other and never both.

    The rule used here is where a load sits relative to the last synchronising event:

    * A load whose address has already been read since the last barrier, acquire or
      exclusive is a re-execution - a loop - and reads current memory. Spins progress.
    * A load reading somewhere new since that point is one the hardware was free to have
      issued at any time after the barrier, so it is answered as of the barrier. This is
      the worst case the architecture permits, and it is what a speculated load gets.

    This is a heuristic standing in for out-of-order execution, not a model of it. It
    encodes the guarantee rather than the mechanism: without a barrier between two loads
    you have established nothing about their order, so the model assumes the worst.
    """

    def __init__(self, model=MemoryModel.SEQUENTIAL):
        self.model = model
        #: Global time of the last synchronising event on this core.
        self.sync_time = 0
        #: Addresses already read since then, at their granularity.
        self.seen = set()
        #: Counters, for showing that reordering actually occurred.
        self.stale_reads = 0
        self.fresh_reads = 0

    @property
    def reordering(self) -> bool:
        # Only the adversarial policy reorders loads. The relaxed policy models a store
        # buffer alone, which is the weaker and more familiar of the two behaviours.
        return self.model is MemoryModel.ADVERSARIAL

    def synchronize(self, now):
        """
        A barrier, acquire or exclusive access: everything after this is ordered against
        everything before it, so the window starts again.
        """
        self.sync_time = now
        self.seen.clear()

    def read_time(self, address, now):
        """
        The effective time at which a load of ``address`` should be answered.
        """
        if not self.reordering:
            return now
        if address in self.seen:
            self.fresh_reads += 1
            return now
        self.seen.add(address)
        self.stale_reads += 1
        return self.sync_time

    def clear(self):
        self.sync_time = 0
        self.seen.clear()
