"""
Capturing and representing MMIO register traces.

A trace is an ordered list of register accesses taken from real hardware.
Replaying one against a peripheral model and comparing the read values is
the strongest correctness check available short of owning the silicon: it
validates the model against what the hardware actually did, rather than
against our reading of a datasheet.

CAPTURING A TRACE ON A REAL PI
------------------------------
Linux has MMIO tracepoints on arm64 behind ``CONFIG_TRACE_MMIO_ACCESS``.
On a Pi 4 running a 64-bit kernel built with that option::

    # cd /sys/kernel/tracing
    # echo 1 > events/rwmmio/enable
    # echo > trace
    ... exercise the driver, e.g. spidev_test or a gpio toggle ...
    # cat trace > /tmp/capture.txt

Each line looks like::

    kworker/0:1-23 [000] ..... 123.456789: rwmmio_write: \
bcm2835_spi_transfer_one+0x1c/0x2f0 width=32 val=0x80 addr=0xffff800008a04000

:func:`parse_ftrace` handles that.  ``rwmmio_read`` records the access and
``rwmmio_post_read`` carries the value that came back, so reads are paired
up during parsing.

If the kernel lacks those tracepoints, a userspace capture via ``/dev/mem``
works too; write it in the canonical format (see :func:`parse_canonical`),
which is just ``op width addr value`` per line.

ADDRESSES
---------
Traces record whatever address the driver used -- usually a kernel virtual
address from ``ioremap``.  Use :meth:`Trace.rebase` to map it onto the
board's physical layout before replaying.
"""

import re
from collections import Counter, namedtuple

#: One register access.  ``op`` is 'r' or 'w'; ``width`` is in bits;
#: ``value`` is what was written, or what a read returned; ``caller`` is the
#: kernel symbol if the trace carried one.
Access = namedtuple('Access', ['op', 'addr', 'width', 'value', 'caller'])
Access.__new__.__defaults__ = (None,)


# ftrace rwmmio lines.  The leading task/cpu/timestamp preamble varies, so
# anchor on the event name rather than trying to match the whole line.
_FTRACE_LINE = re.compile(
    r'rwmmio_(?P<event>write|read|post_read|post_write)\s*:\s*'
    r'(?P<caller>\S+)?\s*'
    r'width=(?P<width>\d+)\s*'
    r'(?:val=(?P<val>0x[0-9a-fA-F]+|\d+)\s*)?'
    r'addr=(?P<addr>0x[0-9a-fA-F]+|\d+)'
)

_CANONICAL_LINE = re.compile(
    r'^\s*(?P<op>[rw])\s+'
    r'(?P<width>\d+)\s+'
    r'(?P<addr>0x[0-9a-fA-F]+|\d+)\s+'
    r'(?P<value>0x[0-9a-fA-F]+|\d+)'
    r'(?:\s+(?P<caller>\S+))?\s*$'
)


def _to_int(text):
    if text is None:
        return None
    text = text.strip()
    return int(text, 16) if text.lower().startswith('0x') else int(text)


class Trace:
    """
    An ordered sequence of :class:`Access` records.

    :param accesses: the records
    :param name: label used in reports
    :param source: where the trace came from, for provenance in reports
    """

    def __init__(self, accesses=None, name='trace', source=None):
        self.accesses = list(accesses or [])
        self.name = name
        #: Free-form provenance string.  Reports print this, because whether
        #: a trace came from real silicon or from our own model is the single
        #: most important thing to know when reading the result.
        self.source = source or 'unspecified'

    def __len__(self):
        return len(self.accesses)

    def __iter__(self):
        return iter(self.accesses)

    def __getitem__(self, index):
        return self.accesses[index]

    # ------------------------------------------------------------------
    def rebase(self, captured_base, target_base, span=0x1000):
        """
        Translate addresses from the captured window onto the board's map.

        A driver traced on real hardware reports kernel virtual addresses;
        ``captured_base`` is where its register block appeared, and
        ``target_base`` is where the same block sits on the emulated board.
        Accesses outside ``span`` bytes of ``captured_base`` are dropped, so
        a trace covering several peripherals can be split per block.
        """
        moved = []
        for access in self.accesses:
            offset = access.addr - captured_base
            if 0 <= offset < span:
                moved.append(access._replace(addr=target_base + offset))
        return Trace(moved, name=self.name, source=self.source)

    def filter(self, predicate):
        """A new trace containing only accesses satisfying ``predicate``."""
        return Trace([a for a in self.accesses if predicate(a)],
                     name=self.name, source=self.source)

    def offsets(self, base):
        """The set of register offsets this trace touches, relative to ``base``."""
        return {a.addr - base for a in self.accesses}

    def summary(self):
        """Counts of reads and writes per offset, for coverage reporting."""
        return Counter((a.op, a.addr) for a in self.accesses)

    # ------------------------------------------------------------------
    def to_canonical(self):
        """Serialise to the canonical text format."""
        lines = [f'# {self.name}', f'# source: {self.source}']
        for a in self.accesses:
            line = f'{a.op} {a.width} {a.addr:#x} {a.value:#x}'
            if a.caller:
                line += f' {a.caller}'
            lines.append(line)
        return '\n'.join(lines) + '\n'


def parse_ftrace(text, name='ftrace', source=None):
    """
    Parse Linux ``rwmmio`` tracepoint output.

    ``rwmmio_read`` events carry no value -- the value arrives on the
    following ``rwmmio_post_read``.  They are paired here, and a read whose
    post event is missing (buffer truncated mid-pair) is dropped rather than
    recorded with a bogus value.
    """
    accesses = []
    pending_read = None
    for line in text.splitlines():
        match = _FTRACE_LINE.search(line)
        if not match:
            continue
        event = match.group('event')
        addr = _to_int(match.group('addr'))
        width = int(match.group('width'))
        value = _to_int(match.group('val'))
        caller = match.group('caller')

        if event == 'write':
            accesses.append(Access('w', addr, width, value or 0, caller))
        elif event == 'read':
            pending_read = (addr, width, caller)
        elif event == 'post_read':
            if pending_read and pending_read[0] == addr:
                accesses.append(Access('r', addr, width, value or 0,
                                       pending_read[2] or caller))
                pending_read = None
            else:
                # Unpaired post_read: still usable, it carries the value.
                accesses.append(Access('r', addr, width, value or 0, caller))
        # post_write carries no extra information for our purposes.
    return Trace(accesses, name=name,
                 source=source or 'ftrace rwmmio capture')


def parse_canonical(text, name='trace', source=None):
    """
    Parse the canonical format: ``op width addr value [caller]`` per line.

    Lines starting with ``#`` are comments; a ``# source:`` comment sets the
    trace's provenance if none is passed in.
    """
    accesses = []
    embedded_source = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            if stripped.lower().startswith('# source:'):
                embedded_source = stripped.split(':', 1)[1].strip()
            continue
        match = _CANONICAL_LINE.match(stripped)
        if not match:
            raise ValueError(f'unparsable trace line: {line!r}')
        accesses.append(Access(
            match.group('op'),
            _to_int(match.group('addr')),
            int(match.group('width')),
            _to_int(match.group('value')),
            match.group('caller'),
        ))
    return Trace(accesses, name=name,
                 source=source or embedded_source or 'canonical capture')


def load(path, name=None, source=None):
    """
    Read a trace file, choosing the parser by content.

    ftrace output is detected by the presence of ``rwmmio_`` events.
    """
    with open(path) as handle:
        text = handle.read()
    label = name or str(path).rsplit('/', 1)[-1]
    if 'rwmmio_' in text:
        return parse_ftrace(text, name=label, source=source)
    return parse_canonical(text, name=label, source=source)


class TraceRecorder:
    """
    Records accesses from a live model into a :class:`Trace`.

    Attach to a device to capture what our own firmware does.  The result is
    useful as a *regression* baseline, but note the provenance: a trace
    recorded from the model and replayed against the model proves only that
    behaviour has not changed, never that it matches hardware.
    """

    def __init__(self, device, base, name='recorded'):
        self.device = device
        self.base = base
        self.name = name
        device.trace = True
        self._start = len(device.accesses)

    def trace(self):
        """Snapshot everything recorded since construction."""
        captured = self.device.accesses[self._start:]
        return Trace(
            [Access('w' if a.kind == 'w' else 'r',
                    self.base + a.offset, a.size * 8, a.value, a.name)
             for a in captured],
            name=self.name,
            source='recorded from armulator model (NOT hardware)',
        )
