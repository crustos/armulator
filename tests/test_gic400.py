import pytest

from armulator.boards import RaspberryPi4
from armulator.peripherals.gic400 import (
    GICC_CTLR, GICC_EOIR, GICC_HPPIR, GICC_IAR, GICC_PMR, GICC_RPR,
    GICD_CTLR, GICD_ICENABLER, GICD_ICFGR, GICD_ICPENDR, GICD_IPRIORITYR,
    GICD_ISENABLER, GICD_ISPENDR, GICD_ITARGETSR, GICD_SGIR, GICD_TYPER,
    SPI_BASE, SPURIOUS_ID, Gic400,
)


@pytest.fixture
def gic():
    """A GIC with the distributor and CPU interface enabled, nothing masked."""
    g = Gic400()
    g.write_register(GICD_CTLR, 1)
    g.write_register(GICC_CTLR, 1)
    g.write_register(GICC_PMR, 0xFF)
    return g


def enable(gic, intid):
    gic.write_register(GICD_ISENABLER + 4 * (intid // 32), 1 << (intid % 32))


def set_edge(gic, intid):
    index = intid // 16
    gic.write_register(GICD_ICFGR + 4 * index, 0b10 << (2 * (intid % 16)))


class TestDistributor:

    def test_disabled_distributor_blocks_everything(self):
        g = Gic400()
        g.write_register(GICC_CTLR, 1)
        enable(g, 40)
        g.set_line(40, True)
        assert g.highest_priority_pending() is None
        assert g.irq_pending is False

    def test_interrupt_must_be_enabled(self, gic):
        gic.set_line(40, True)
        assert gic.highest_priority_pending() is None
        enable(gic, 40)
        assert gic.highest_priority_pending() == 40

    def test_icenabler_disables(self, gic):
        enable(gic, 40)
        gic.set_line(40, True)
        assert gic.irq_pending is True
        gic.write_register(GICD_ICENABLER + 4, 1 << 8)     # intid 40
        assert gic.irq_pending is False

    def test_isenabler_reads_back_enable_state(self, gic):
        enable(gic, 40)
        assert gic.read_register(GICD_ISENABLER + 4) & (1 << 8)

    def test_software_can_force_pending(self, gic):
        enable(gic, 50)
        gic.write_register(GICD_ISPENDR + 4, 1 << 18)      # intid 50
        assert gic.highest_priority_pending() == 50

    def test_software_can_clear_pending(self, gic):
        enable(gic, 50)
        set_edge(gic, 50)
        gic.set_line(50, True)
        assert gic.highest_priority_pending() == 50
        gic.write_register(GICD_ICPENDR + 4, 1 << 18)
        assert gic.highest_priority_pending() is None

    def test_typer_reports_interrupt_count(self, gic):
        # ITLinesNumber encodes (N+1)*32 supported interrupts.
        typer = gic.read_register(GICD_TYPER)
        assert ((typer & 0x1F) + 1) * 32 == gic.num_interrupts

    def test_targets_are_readable(self, gic):
        gic.write_register(GICD_ITARGETSR + 4 * 10, 0x02020202)
        assert gic.read_register(GICD_ITARGETSR + 4 * 10) == 0x02020202


class TestPriority:

    def test_lower_value_wins(self, gic):
        enable(gic, 40)
        enable(gic, 41)
        gic.priority[40] = 0x80
        gic.priority[41] = 0x40                            # higher priority
        gic.set_line(40, True)
        gic.set_line(41, True)
        assert gic.highest_priority_pending() == 41

    def test_ties_break_to_lower_id(self, gic):
        enable(gic, 40)
        enable(gic, 41)
        gic.priority[40] = gic.priority[41] = 0x80
        gic.set_line(40, True)
        gic.set_line(41, True)
        assert gic.highest_priority_pending() == 40

    def test_priority_mask_blocks_low_priority(self, gic):
        enable(gic, 40)
        gic.priority[40] = 0x80
        gic.write_register(GICC_PMR, 0x40)                 # only < 0x40 passes
        gic.set_line(40, True)
        assert gic.highest_priority_pending() is None
        gic.write_register(GICC_PMR, 0xF0)
        assert gic.highest_priority_pending() == 40

    def test_priority_register_is_byte_addressed(self, gic):
        gic.write_register(GICD_IPRIORITYR + 4 * 10, 0x11223344)
        assert gic.priority[40] == 0x44
        assert gic.priority[43] == 0x11

    def test_running_priority_reflects_active_interrupt(self, gic):
        enable(gic, 40)
        gic.priority[40] = 0x30
        gic.set_line(40, True)
        gic.read_register(GICC_IAR)
        assert gic.read_register(GICC_RPR) == 0x30


class TestAcknowledgeCycle:

    def test_iar_returns_spurious_when_idle(self, gic):
        assert gic.read_register(GICC_IAR) == SPURIOUS_ID

    def test_acknowledge_moves_to_active(self, gic):
        enable(gic, 40)
        gic.set_line(40, True)
        assert gic.read_register(GICC_IAR) == 40
        assert gic.active[40] is True
        assert gic.irq_pending is False        # no longer signalling the CPU

    def test_hppir_previews_without_acknowledging(self, gic):
        enable(gic, 40)
        gic.set_line(40, True)
        assert gic.read_register(GICC_HPPIR) == 40
        assert gic.active[40] is False         # peek did not consume it

    def test_eoi_deactivates(self, gic):
        enable(gic, 40)
        gic.set_line(40, True)
        intid = gic.read_register(GICC_IAR)
        gic.set_line(40, False)
        gic.write_register(GICC_EOIR, intid)
        assert gic.active[40] is False
        assert gic.irq_pending is False

    def test_active_interrupt_does_not_re_present(self, gic):
        enable(gic, 40)
        gic.set_line(40, True)
        gic.read_register(GICC_IAR)
        assert gic.read_register(GICC_IAR) == SPURIOUS_ID

    def test_level_source_still_asserting_re_fires_after_eoi(self, gic):
        # This is why a handler must clear the device before EOI.
        enable(gic, 40)
        gic.set_line(40, True)
        intid = gic.read_register(GICC_IAR)
        gic.write_register(GICC_EOIR, intid)   # line never deasserted
        assert gic.irq_pending is True
        assert gic.read_register(GICC_IAR) == 40

    def test_edge_source_does_not_re_fire_after_eoi(self, gic):
        enable(gic, 40)
        set_edge(gic, 40)
        gic.set_line(40, True)
        intid = gic.read_register(GICC_IAR)
        gic.write_register(GICC_EOIR, intid)
        assert gic.irq_pending is False

    def test_higher_priority_preempts_while_lower_active(self, gic):
        enable(gic, 40)
        enable(gic, 41)
        gic.priority[40] = 0x80
        gic.priority[41] = 0x10
        gic.set_line(40, True)
        assert gic.read_register(GICC_IAR) == 40
        gic.set_line(41, True)                 # higher priority arrives
        assert gic.read_register(GICC_IAR) == 41


class TestTriggerModes:

    def test_edge_latches_a_transient_pulse(self, gic):
        enable(gic, 40)
        set_edge(gic, 40)
        gic.set_line(40, True)
        gic.set_line(40, False)                # gone before anyone looked
        assert gic.highest_priority_pending() == 40

    def test_level_does_not_latch_a_transient_pulse(self, gic):
        enable(gic, 40)
        gic.set_line(40, True)
        gic.set_line(40, False)
        assert gic.highest_priority_pending() is None

    def test_icfgr_reads_back_configuration(self, gic):
        set_edge(gic, 40)
        value = gic.read_register(GICD_ICFGR + 4 * (40 // 16))
        assert value & (0b10 << (2 * (40 % 16)))


class TestSoftwareGeneratedInterrupts:

    def test_sgi_via_register(self, gic):
        enable(gic, 3)
        gic.write_register(GICD_SGIR, 3 | (0x01 << 16))
        assert gic.highest_priority_pending() == 3

    def test_sgi_helper_validates_id(self, gic):
        with pytest.raises(ValueError):
            gic.send_sgi(16)


class TestBoardIntegration:

    def test_pi4_wires_devices_to_documented_spis(self):
        board = RaspberryPi4()
        assert board.gic is not None
        ids = set(board.gic.sources)
        assert SPI_BASE + RaspberryPi4.GPIO_SPI in ids
        assert SPI_BASE + RaspberryPi4.UART_SPI in ids

    def test_gic_mapped_outside_peripheral_window(self):
        board = RaspberryPi4()
        bases = {mc.mem: mc.beginning for mc in board.cpu.mem.memories}
        assert bases[board.gic] == 0xFF840000

    def test_device_line_reaches_distributor_via_refresh(self):
        board = RaspberryPi4()
        gic = board.gic
        intid = SPI_BASE + RaspberryPi4.GPIO_SPI
        gic.write_register(GICD_CTLR, 1)
        gic.write_register(GICC_CTLR, 1)
        gic.write_register(GICC_PMR, 0xFF)
        enable(gic, intid)
        board.gpio.write_register(0x4C, 1 << 7)        # GPREN0 pin 7
        board.gpio.drive_input(7, True)
        gic.refresh()
        assert gic.highest_priority_pending() == intid

    def test_masked_gic_does_not_interrupt_cpu(self):
        board = RaspberryPi4()
        board.gic.write_register(GICD_CTLR, 0)         # distributor off
        board.cpu.registers.cpsr.i = 0
        board.gpio.write_register(0x4C, 1 << 7)
        board.gpio.drive_input(7, True)
        assert board.service_interrupts() is False

    def test_enabled_gic_interrupts_cpu(self):
        board = RaspberryPi4()
        gic = board.gic
        intid = SPI_BASE + RaspberryPi4.GPIO_SPI
        gic.write_register(GICD_CTLR, 1)
        gic.write_register(GICC_CTLR, 1)
        gic.write_register(GICC_PMR, 0xFF)
        enable(gic, intid)
        board.cpu.registers.cpsr.i = 0
        board.start()
        board.gpio.write_register(0x4C, 1 << 7)
        board.gpio.drive_input(7, True)
        assert board.service_interrupts() is True
        assert board.cpu.registers.cpsr.m == 0b10010   # IRQ mode

    def test_board_without_gic_still_polls_devices(self):
        from armulator.boards import RaspberryPi3
        board = RaspberryPi3()
        assert board.gic is None
        board.cpu.registers.cpsr.i = 0
        board.start()
        board.gpio.write_register(0x4C, 1 << 7)
        board.gpio.drive_input(7, True)
        assert board.service_interrupts() is True

    def test_connect_irq_without_gic_raises(self):
        from armulator.boards import RaspberryPi3
        board = RaspberryPi3()
        with pytest.raises(RuntimeError):
            board.connect_irq(board.gpio, 10)
