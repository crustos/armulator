"""
Validating a peripheral model against a captured driver trace.

Run with:  python3 example/replay_driver_trace.py

The replay harness is the strongest correctness check available without
owning the hardware: apply a real driver's writes to the model, and assert
every read returns what the silicon returned.

Read scenario 4 before trusting any of this. The distinction between a trace
captured from hardware and one recorded from the model is the difference
between validation and circular reasoning, and the harness is built to keep
that distinction visible.
"""

from armulator.boards import RaspberryPi4
from armulator.harness import (
    Access, Trace, TraceRecorder, parse_ftrace, replay, replay_on_board,
)
from armulator.peripherals.gpio_bcm import BcmGpio

# An ftrace capture in the format a Pi 4 kernel built with
# CONFIG_TRACE_MMIO_ACCESS emits.  Hand-written here for illustration --
# see scenario 4 and RASPI.md on getting the real thing.
CAPTURE = """\
# tracer: nop
#
    kworker/0:1-23  [000] .....  91.100010: rwmmio_write: \
bcm2835_gpio_direction_output+0x24/0x60 width=32 val=0x200000 addr=0xffff800008a00004
    kworker/0:1-23  [000] .....  91.100021: rwmmio_write: \
bcm2835_gpio_set+0x18/0x40 width=32 val=0x20000 addr=0xffff800008a0001c
    kworker/0:1-23  [000] .....  91.100032: rwmmio_read: \
bcm2835_gpio_get+0x14/0x38 width=32 addr=0xffff800008a00034
    kworker/0:1-23  [000] .....  91.100034: rwmmio_post_read: \
bcm2835_gpio_get+0x14/0x38 width=32 val=0x20000 addr=0xffff800008a00034
"""

GPIO_CAPTURED_BASE = 0xFFFF800008A00000


def scenario_1_replay_a_capture():
    """Apply a driver's register sequence and check the model agrees."""
    trace = parse_ftrace(CAPTURE, name='gpio_direction_output')
    board = RaspberryPi4()
    report = replay_on_board(board, 'gpio', trace, GPIO_CAPTURED_BASE)

    print('1. replaying a driver capture')
    print(report.format())
    assert report.ok, 'model diverged from the captured behaviour'


def scenario_2_a_divergence_is_caught():
    """
    What a real modelling bug looks like.

    Here the trace says GPLEV0 read back 0x20000 (pin 17 high), but the
    driver never configured pin 17 as an output first -- so a model that
    got the FSEL gating wrong would return 0. The harness names the
    register, the expected and actual values, and the kernel function.
    """
    trace = Trace([
        # Note: no GPFSEL write. Pin 17 is still an input.
        Access('w', 0x1C, 32, 1 << 17, 'bcm2835_gpio_set+0x18/0x40'),
        Access('r', 0x34, 32, 1 << 17, 'bcm2835_gpio_get+0x14/0x38'),
    ], name='missing_fsel', source='illustrative, hand-written')

    report = replay(BcmGpio(pull_style='bcm2711'), trace, base=0)
    print('\n2. a divergence, reported')
    print(report.format())
    assert not report.ok
    assert report.mismatches[0].register == 'GPLEV0'


def scenario_3_volatile_registers():
    """
    Not every read can match, and pretending otherwise is worse than useless.

    A free-running counter returns a different value every read; comparing
    it against a capture guarantees a false failure. Declaring it volatile
    executes the read but skips the comparison, and the report says how many
    were skipped so the gap stays visible.
    """
    board = RaspberryPi4()
    trace = Trace([
        Access('r', 0xFE003004, 32, 0x12345678, 'bcm2835_time_get+0x8/0x20'),
    ], name='timer_read', source='illustrative, hand-written')

    strict = replay(board.timer, trace, base=0xFE003000)
    lenient = replay(board.timer, trace, base=0xFE003000, volatile={0x04})

    print('\n3. volatile registers')
    print(f'   compared strictly -> {"PASS" if strict.ok else "FAIL"} '
          f'({len(strict.mismatches)} mismatch)')
    print(f'   CLO marked volatile -> {"PASS" if lenient.ok else "FAIL"} '
          f'({lenient.skipped_volatile} read skipped, not compared)')
    assert not strict.ok and lenient.ok


def scenario_4_provenance_matters():
    """
    The trap this harness is designed to make impossible to fall into.

    Recording a trace from the model and replaying it against the model
    always passes. It is a useful regression guard and a worthless
    validation. The report says so, unprompted, every time.
    """
    source = BcmGpio(pull_style='bcm2711')
    recorder = TraceRecorder(source, 0xFE200000, name='self_recorded')
    source.write(0x04, 4, 0b001 << 21)
    source.write(0x1C, 4, 1 << 17)
    source.read(0x34, 4)

    report = replay(BcmGpio(pull_style='bcm2711'),
                    recorder.trace().rebase(0xFE200000, 0), base=0)

    print('\n4. provenance')
    print(report.format())
    assert report.ok
    assert 'regression check only' in report.format()


def scenario_5_coverage():
    """
    A passing replay only covers what the trace touched.

    Coverage is reported alongside the result, because "PASS" on a trace
    that exercised three of twenty registers is a much weaker claim than it
    appears.
    """
    trace = parse_ftrace(CAPTURE)
    board = RaspberryPi4()
    report = replay_on_board(board, 'gpio', trace, GPIO_CAPTURED_BASE)

    touched = report.registers_touched()
    untouched = report.untouched_registers()
    print('\n5. coverage')
    print(f'   exercised    : {", ".join(touched)}')
    print(f'   not exercised: {len(untouched)} registers')
    print(f'   -> this PASS covers {len(touched)} of '
          f'{len(touched) + len(untouched)} documented registers')
    assert touched and untouched


if __name__ == '__main__':
    scenario_1_replay_a_capture()
    scenario_2_a_divergence_is_caught()
    scenario_3_volatile_registers()
    scenario_4_provenance_matters()
    scenario_5_coverage()
    print('\nAll scenarios passed.')
