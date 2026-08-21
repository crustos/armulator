"""
Replaying hardware traces against peripheral models.

The idea is simple and the value is high: take a trace of what a real driver
did to real silicon, apply the writes to our model, and check that every
read returns what the hardware returned.  Any divergence is a modelling bug,
found without owning the board.

    report = replay(board.spi, trace, base=0xFE204000)
    assert report.ok, report.format()

WHAT A PASS DOES AND DOES NOT PROVE
-----------------------------------
A pass means the model reproduced the hardware's read values for the
registers the trace exercised, in the order the driver touched them.  It
says nothing about registers the trace never touched -- which is why
:class:`ReplayReport` reports coverage alongside correctness.

Critically, a trace recorded *from the model* and replayed against the model
is circular and proves only that behaviour has not regressed.  Reports carry
the trace's provenance so this is visible in the output rather than being
something you have to remember.

VOLATILE REGISTERS
------------------
Some reads legitimately differ between hardware and model: free-running
counters, FIFO occupancy that depends on real timing, status bits reflecting
bus activity we do not simulate.  Those are declared per-replay rather than
silently tolerated::

    replay(board.timer, trace, base=..., volatile={0x04, 0x08})

Anything volatile is executed but not compared, and is listed separately in
the report so the untested surface stays visible.
"""

from collections import Counter, namedtuple

#: A single divergence between the model and the trace.
Mismatch = namedtuple(
    'Mismatch', ['index', 'offset', 'register', 'expected', 'actual', 'caller']
)


class ReplayReport:
    """Outcome of replaying a trace against a device model."""

    def __init__(self, device, trace, base):
        self.device = device
        self.trace = trace
        self.base = base
        self.mismatches = []
        self.compared = 0
        self.writes = 0
        self.skipped_volatile = 0
        self.out_of_range = 0
        #: Offsets touched by the trace that the model does not implement.
        self.unimplemented = set()
        #: Read/write counts per offset, for coverage.
        self.coverage = Counter()

    @property
    def ok(self):
        return not self.mismatches and not self.out_of_range

    @property
    def total(self):
        return self.compared + self.writes + self.skipped_volatile

    def registers_touched(self):
        """Register names the trace exercised, in offset order."""
        return [self.device.register_name(o)
                for o in sorted({o for _, o in self.coverage})]

    def untouched_registers(self):
        """
        Documented registers the trace never exercised.

        This is the honest measure of what a passing replay leaves unproven.
        """
        touched = {o for _, o in self.coverage}
        declared = set(getattr(self.device, 'REGISTERS', {}))
        return sorted(declared - touched)

    def format(self):
        lines = [
            f'Replay of {self.trace.name!r} against {self.device.name}',
            f'  provenance : {self.trace.source}',
            f'  accesses   : {self.total} '
            f'({self.writes} writes, {self.compared} reads compared, '
            f'{self.skipped_volatile} volatile reads skipped)',
            f'  result     : {"PASS" if self.ok else "FAIL"}',
        ]
        if self.out_of_range:
            lines.append(
                f'  WARNING: {self.out_of_range} accesses fell outside the '
                f'device window at {self.base:#x} -- wrong base or span?'
            )
        if self.unimplemented:
            names = ', '.join(f'+{o:#05x}' for o in sorted(self.unimplemented))
            lines.append(f'  unimplemented offsets touched: {names}')
        if self.mismatches:
            lines.append(f'  {len(self.mismatches)} mismatch(es):')
            for m in self.mismatches[:20]:
                caller = f'  [{m.caller}]' if m.caller else ''
                lines.append(
                    f'    #{m.index} {m.register} (+{m.offset:#05x}): '
                    f'hardware={m.expected:#010x} model={m.actual:#010x}{caller}'
                )
            if len(self.mismatches) > 20:
                lines.append(f'    ... and {len(self.mismatches) - 20} more')
        untouched = self.untouched_registers()
        if untouched:
            names = ', '.join(
                self.device.register_name(o) for o in untouched[:12]
            )
            more = '' if len(untouched) <= 12 else f' (+{len(untouched) - 12} more)'
            lines.append(f'  not exercised by this trace: {names}{more}')
        if 'NOT hardware' in self.trace.source:
            lines.append(
                '  NOTE: this trace came from the model, so a pass is a '
                'regression check only -- it does not validate against silicon.'
            )
        return '\n'.join(lines)

    def __str__(self):
        return self.format()


def replay(device, trace, base, volatile=None, strict_writes=False,
           reset_between=False):
    """
    Replay ``trace`` against ``device`` and report divergences.

    :param device: an :class:`~armulator.peripherals.mmio.MMIODevice`
    :param trace: a :class:`~armulator.harness.trace.Trace`
    :param base: physical address the device is mapped at, used to turn
        trace addresses into register offsets
    :param volatile: offsets whose read values should be executed but not
        compared (counters, timing-dependent status)
    :param strict_writes: also verify that writes land where expected by
        reading the register back; off by default because many registers are
        write-only or self-clearing strobes, where read-back legitimately
        differs
    :param reset_between: unused placeholder for future per-access reset
    :returns: a :class:`ReplayReport`
    """
    volatile = set(volatile or ())
    report = ReplayReport(device, trace, base)
    declared = set(getattr(device, 'REGISTERS', {}))

    for index, access in enumerate(trace):
        offset = access.addr - base
        if offset < 0 or offset >= device.size:
            report.out_of_range += 1
            continue

        word_offset = offset & ~0x3
        report.coverage[(access.op, word_offset)] += 1
        if declared and word_offset not in declared:
            report.unimplemented.add(word_offset)

        size = max(1, access.width // 8)

        if access.op == 'w':
            device.write(offset, size, access.value)
            report.writes += 1
            if strict_writes and word_offset not in volatile:
                read_back = int.from_bytes(device.read(offset, size), 'little')
                if read_back != access.value:
                    report.mismatches.append(Mismatch(
                        index, word_offset, device.register_name(word_offset),
                        access.value, read_back, access.caller,
                    ))
            continue

        actual = int.from_bytes(device.read(offset, size), 'little')
        if word_offset in volatile:
            report.skipped_volatile += 1
            continue
        report.compared += 1
        if actual != access.value:
            report.mismatches.append(Mismatch(
                index, word_offset, device.register_name(word_offset),
                access.value, actual, access.caller,
            ))

    return report


def replay_on_board(board, device_name, trace, captured_base, volatile=None,
                    **kwargs):
    """
    Convenience wrapper: rebase a trace onto a board device and replay it.

    ``captured_base`` is where the register block appeared in the capture;
    the device's mapped address on the board is looked up automatically.
    """
    device = board.devices[device_name]
    target_base = next(
        mc.beginning for mc in board.cpu.mem.memories if mc.mem is device
    )
    rebased = trace.rebase(captured_base, target_base, span=device.size)
    return replay(device, rebased, target_base, volatile=volatile, **kwargs)
