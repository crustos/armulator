from pathlib import Path

import pytest

from armulator.boards import JetsonNano, RaspberryPi3, RaspberryPi4
from armulator.harness import (
    Access, Trace, TraceRecorder, load, parse_canonical, parse_ftrace, replay,
    replay_on_board,
)
from armulator.peripherals.gpio_bcm import BcmGpio
from armulator.peripherals.uart_pl011 import BcmSystemTimer

TRACES = Path(__file__).resolve().parent.parent / 'traces'

# A capture as a Pi 4 kernel with CONFIG_TRACE_MMIO_ACCESS would emit it.
FTRACE_SAMPLE = """\
# tracer: nop
#
     kworker/0:1-23    [000] .....   123.456789: rwmmio_write: \
bcm2835_spi_transfer_one+0x1c/0x2f0 width=32 val=0x80 addr=0xffff800008a04000
     kworker/0:1-23    [000] .....   123.456800: rwmmio_write: \
bcm2835_spi_transfer_one+0x40/0x2f0 width=32 val=0x42 addr=0xffff800008a04004
     kworker/0:1-23    [000] .....   123.456810: rwmmio_read: \
bcm2835_spi_transfer_one+0x60/0x2f0 width=32 addr=0xffff800008a04000
     kworker/0:1-23    [000] .....   123.456812: rwmmio_post_read: \
bcm2835_spi_transfer_one+0x60/0x2f0 width=32 val=0xe0080 addr=0xffff800008a04000
"""


class TestFtraceParsing:

    def test_parses_writes_and_reads(self):
        trace = parse_ftrace(FTRACE_SAMPLE)
        assert len(trace) == 3
        assert [a.op for a in trace] == ['w', 'w', 'r']

    def test_pairs_read_with_post_read(self):
        trace = parse_ftrace(FTRACE_SAMPLE)
        read = trace[2]
        assert read.op == 'r'
        assert read.value == 0xE0080          # value comes from post_read

    def test_captures_caller_symbol(self):
        trace = parse_ftrace(FTRACE_SAMPLE)
        assert 'bcm2835_spi_transfer_one' in trace[0].caller

    def test_captures_width(self):
        assert all(a.width == 32 for a in parse_ftrace(FTRACE_SAMPLE))

    def test_ignores_comment_and_header_lines(self):
        assert len(parse_ftrace('# tracer: nop\n#\n')) == 0

    def test_unpaired_read_is_dropped(self):
        # A read with no post_read carries no value; recording it as zero
        # would silently inject a false expectation.
        text = ('  x-1 [000] ..... 1.0: rwmmio_read: fn+0x0/0x1 '
                'width=32 addr=0x1000\n')
        assert len(parse_ftrace(text)) == 0

    def test_unpaired_post_read_is_still_used(self):
        text = ('  x-1 [000] ..... 1.0: rwmmio_post_read: fn+0x0/0x1 '
                'width=32 val=0x5 addr=0x1000\n')
        trace = parse_ftrace(text)
        assert len(trace) == 1 and trace[0].value == 5

    def test_provenance_recorded(self):
        assert 'ftrace' in parse_ftrace(FTRACE_SAMPLE).source


class TestCanonicalFormat:

    def test_round_trip(self):
        original = parse_ftrace(FTRACE_SAMPLE)
        again = parse_canonical(original.to_canonical())
        assert again.accesses == original.accesses

    def test_source_comment_is_read_back(self):
        text = '# source: my bench capture\nw 32 0x1000 0x1\n'
        assert parse_canonical(text).source == 'my bench capture'

    def test_decimal_and_hex_both_parse(self):
        trace = parse_canonical('w 32 4096 255\nw 32 0x1000 0xff\n')
        assert trace[0] == trace[1]._replace(caller=None)

    def test_bad_line_raises(self):
        with pytest.raises(ValueError):
            parse_canonical('this is not a trace line\n')

    def test_blank_lines_ignored(self):
        assert len(parse_canonical('\n\nw 32 0x1000 0x1\n\n')) == 1


class TestRebase:

    def test_maps_onto_board_addresses(self):
        trace = parse_ftrace(FTRACE_SAMPLE)
        moved = trace.rebase(0xFFFF800008A04000, 0xFE204000)
        assert [a.addr for a in moved] == [0xFE204000, 0xFE204004, 0xFE204000]

    def test_drops_accesses_outside_the_window(self):
        trace = Trace([
            Access('w', 0x1000, 32, 1),
            Access('w', 0x9999, 32, 1),       # far outside
        ])
        assert len(trace.rebase(0x1000, 0x2000, span=0x100)) == 1

    def test_preserves_provenance(self):
        trace = parse_ftrace(FTRACE_SAMPLE)
        assert trace.rebase(0xFFFF800008A04000, 0xFE204000).source == trace.source


class TestReplay:

    def test_matching_trace_passes(self):
        gpio = BcmGpio(pull_style='bcm2711')
        trace = Trace([
            Access('w', 0x04, 32, 0b001 << 21),
            Access('w', 0x1C, 32, 1 << 17),
            Access('r', 0x34, 32, 1 << 17),        # GPLEV0 reads pin 17 high
        ])
        report = replay(gpio, trace, base=0)
        assert report.ok, report.format()
        assert report.compared == 1
        assert report.writes == 2

    def test_divergent_read_is_reported(self):
        gpio = BcmGpio(pull_style='bcm2711')
        trace = Trace([
            Access('w', 0x04, 32, 0b001 << 21),
            Access('r', 0x34, 32, 0xDEADBEEF),    # not what the model returns
        ])
        report = replay(gpio, trace, base=0)
        assert not report.ok
        assert report.mismatches[0].expected == 0xDEADBEEF
        assert report.mismatches[0].register == 'GPLEV0'

    def test_mismatch_carries_caller(self):
        gpio = BcmGpio()
        trace = Trace([Access('r', 0x34, 32, 0xFF, 'gpio_get_value+0x10')])
        report = replay(gpio, trace, base=0)
        assert 'gpio_get_value' in report.mismatches[0].caller

    def test_volatile_registers_are_skipped(self):
        timer = BcmSystemTimer(auto_advance=1)
        trace = Trace([Access('r', 0x04, 32, 999999)])   # free-running counter
        strict = replay(timer, trace, base=0)
        lenient = replay(timer, trace, base=0, volatile={0x04})
        assert not strict.ok
        assert lenient.ok
        assert lenient.skipped_volatile == 1

    def test_out_of_range_access_is_flagged(self):
        gpio = BcmGpio()
        trace = Trace([Access('w', 0x99999, 32, 1)])
        report = replay(gpio, trace, base=0)
        assert report.out_of_range == 1
        assert not report.ok

    def test_unimplemented_offset_is_reported(self):
        gpio = BcmGpio()
        trace = Trace([Access('w', 0x300, 32, 1)])   # not a documented register
        report = replay(gpio, trace, base=0)
        assert 0x300 in report.unimplemented

    def test_strict_writes_catches_readback_divergence(self):
        gpio = BcmGpio()
        # GPSET0 is write-only: it reads back as zero, so a strict read-back
        # check must flag it. This is why strict_writes is off by default.
        trace = Trace([Access('w', 0x1C, 32, 1 << 17)])
        assert replay(gpio, trace, base=0).ok
        assert not replay(gpio, trace, base=0, strict_writes=True).ok

    def test_coverage_lists_touched_registers(self):
        gpio = BcmGpio()
        trace = Trace([
            Access('w', 0x04, 32, 0),
            Access('r', 0x34, 32, 0),
        ])
        report = replay(gpio, trace, base=0)
        assert 'GPFSEL1' in report.registers_touched()
        assert 'GPLEV0' in report.registers_touched()

    def test_untouched_registers_are_reported(self):
        gpio = BcmGpio()
        report = replay(gpio, Trace([Access('w', 0x04, 32, 0)]), base=0)
        untouched = {gpio.register_name(o) for o in report.untouched_registers()}
        assert 'GPFSEL0' in untouched
        assert 'GPFSEL1' not in untouched

    def test_byte_width_access(self):
        uart = __import__(
            'armulator.peripherals.uart_pl011', fromlist=['Pl011Uart']
        ).Pl011Uart()
        trace = Trace([Access('w', 0x00, 8, 0x41)])
        report = replay(uart, trace, base=0)
        assert report.ok
        assert uart.tx_buffer == b'A'

    def test_report_format_mentions_provenance(self):
        gpio = BcmGpio()
        trace = Trace([Access('w', 0x04, 32, 0)], source='bench capture')
        assert 'bench capture' in replay(gpio, trace, base=0).format()

    def test_report_warns_on_self_recorded_trace(self):
        gpio = BcmGpio()
        trace = Trace([Access('w', 0x04, 32, 0)],
                      source='recorded from armulator model (NOT hardware)')
        assert 'regression check only' in replay(gpio, trace, base=0).format()


class TestReplayOnBoard:

    def test_rebases_and_replays_against_named_device(self):
        board = RaspberryPi4()
        trace = parse_ftrace(FTRACE_SAMPLE)
        report = replay_on_board(board, 'spi', trace, 0xFFFF800008A04000)
        assert report.total == 3

    def test_unknown_device_raises(self):
        with pytest.raises(KeyError):
            replay_on_board(RaspberryPi4(), 'nosuch', Trace([]), 0)


class TestRecorder:

    def test_records_model_accesses(self):
        gpio = BcmGpio(pull_style='bcm2711')
        recorder = TraceRecorder(gpio, 0xFE200000)
        gpio.write(0x04, 4, 0b001 << 21)
        gpio.read(0x34, 4)
        trace = recorder.trace()
        assert [a.op for a in trace] == ['w', 'r']
        assert trace[0].addr == 0xFE200004

    def test_recorded_trace_is_marked_non_hardware(self):
        gpio = BcmGpio()
        recorder = TraceRecorder(gpio, 0)
        gpio.write(0x04, 4, 0)
        assert 'NOT hardware' in recorder.trace().source

    def test_recorded_trace_replays_cleanly(self):
        # Round-trip: record from a model, replay against a fresh one.
        source = BcmGpio(pull_style='bcm2711')
        recorder = TraceRecorder(source, 0)
        source.write(0x04, 4, 0b001 << 21)
        source.write(0x1C, 4, 1 << 17)
        source.read(0x34, 4)
        trace = recorder.trace()

        report = replay(BcmGpio(pull_style='bcm2711'), trace, base=0)
        assert report.ok, report.format()


class TestBaselineTraces:
    """
    Regression guards.  These traces came from the models, so a pass proves
    behaviour is unchanged -- not that it matches hardware.
    """

    @staticmethod
    def _enable_spi_slave(board):
        from armulator.peripherals.spi_slave import (
            CR_EN, CR_RXE, CR_SPI, CR_TXE, SLV_CR, SLV_SLV,
        )
        board.spi_slave.write_register(SLV_SLV, 0x2A)
        board.spi_slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE)

    @staticmethod
    def _attach_i2c_slave(board):
        from armulator.peripherals.serial_bus import I2cSlaveDevice
        board.i2c.attach_slave(I2cSlaveDevice(address=0x48))

    #: name -> (device, base, board class, volatile offsets, setup hook)
    BASELINES = {
        'gpio_blink': ('gpio', 0xFE200000, RaspberryPi4, set(), None),
        'gpio_pull_bcm2711': ('gpio', 0xFE200000, RaspberryPi4, set(), None),
        'uart_hello': ('uart', 0xFE201000, RaspberryPi4, set(), None),
        'spi_master_transfer': ('spi', 0xFE204000, RaspberryPi4, set(), None),
        'i2c_write': ('i2c', 0xFE804000, RaspberryPi4, set(), '_attach_i2c_slave'),
        'spi_slave_dialogue': ('spi_slave', 0x3F214000, RaspberryPi3, set(),
                               '_enable_spi_slave'),
        'tegra_spi_transfer': ('spi', 0x7000D400, JetsonNano, set(), None),
    }

    @pytest.mark.parametrize('name', sorted(BASELINES))
    def test_baseline_replays_without_divergence(self, name):
        device_name, base, board_cls, volatile, setup = self.BASELINES[name]
        path = TRACES / f'{name}.trace'
        if not path.exists():
            pytest.skip(f'{path.name} not generated; run tools/record_baselines.py')
        trace = load(path)
        board = board_cls()
        if setup:
            getattr(self, setup)(board)
        report = replay(
            board.devices[device_name], trace, base=base, volatile=volatile
        )
        assert report.ok, report.format()

    def test_every_baseline_file_is_covered(self):
        # A generated baseline nobody replays is dead weight.
        on_disk = {p.stem for p in TRACES.glob('*.trace')}
        assert on_disk <= set(self.BASELINES), (
            f'unreplayed baselines: {sorted(on_disk - set(self.BASELINES))}'
        )

    def test_baselines_are_marked_non_hardware(self):
        for path in TRACES.glob('*.trace'):
            assert 'NOT hardware' in load(path).source, path.name
