"""
Board fidelity: per-core interrupt banking, unmapped accesses, timer rates.

Each of these was an approximation that produced plausible-looking results.
A cluster's four timers appeared as one interrupt, an access to an address
with nothing behind it read as zero instead of aborting, and every board
claimed the same counter frequency. None of them raised an error; they just
quietly made the model disagree with hardware.
"""

import pytest

from armulator.armv8.generic_timer import CTL_ENABLE, TIMER_PPI
from armulator.boards import (
    JetsonNano, JetsonNanoA64, JetsonNanoA64Smp, RaspberryPi3, RaspberryPi3A64,
    RaspberryPi4, RaspberryPi4A64,
)
from armulator.peripherals.gic400 import (
    SPI_BASE, BankedInterruptState, Gic400,
)


class TestBankedInterruptState:

    def gic(self):
        return Gic400(name='gic', num_cpus=4)

    def test_ppis_are_private_to_each_core(self):
        gic = self.gic()
        gic.lines.set(TIMER_PPI, True, 2)
        assert gic.lines.get(TIMER_PPI, 2) is True
        for other in (0, 1, 3):
            assert gic.lines.get(TIMER_PPI, other) is False

    def test_spis_are_shared_between_cores(self):
        gic = self.gic()
        spi = SPI_BASE + 5
        gic.lines.set(spi, True, 0)
        # One device drives one line, so every core sees the same SPI.
        for cpu in range(4):
            assert gic.lines.get(spi, cpu) is True

    def test_plain_indexing_follows_the_selected_core(self):
        gic = self.gic()
        gic.current_cpu = 1
        gic.lines[TIMER_PPI] = True
        assert gic.lines.get(TIMER_PPI, 1) is True
        assert gic.lines.get(TIMER_PPI, 0) is False
        gic.current_cpu = 0
        assert gic.lines[TIMER_PPI] is False

    def test_set_all_cpus_drives_every_bank(self):
        gic = self.gic()
        gic.lines.set_all_cpus(TIMER_PPI, True)
        assert all(gic.lines.get(TIMER_PPI, c) for c in range(4))

    def test_length_is_the_interrupt_count(self):
        gic = self.gic()
        assert len(gic.lines) == gic.num_interrupts

    def test_state_is_independent_per_array(self):
        gic = self.gic()
        gic.pending.set(TIMER_PPI, True, 0)
        assert gic.active.get(TIMER_PPI, 0) is False
        assert gic.lines.get(TIMER_PPI, 0) is False

    def test_boundary_interrupt_31_is_banked_and_32_is_not(self):
        gic = self.gic()
        gic.lines.set(SPI_BASE - 1, True, 3)
        assert gic.lines.get(SPI_BASE - 1, 0) is False
        gic.lines.set(SPI_BASE, True, 3)
        assert gic.lines.get(SPI_BASE, 0) is True

    def test_constructed_directly_without_a_gic_attribute_clash(self):
        gic = self.gic()
        state = BankedInterruptState(gic, 64, 2, initial=True)
        assert state.get(0, 0) is True
        assert state.get(SPI_BASE, 1) is True


class TestPerCoreTimers:

    def test_each_core_drives_its_own_timer_ppi(self):
        board = JetsonNanoA64Smp(ram_size=0x2000)
        assert len(board.cores) == 4

        timer = board.cores[2].registers.generic_timer
        timer.tval = 5
        timer.ctl = CTL_ENABLE
        timer.tick(50)
        board.sample_timer()

        assert board.gic.lines.get(TIMER_PPI, 2) is True
        for other in (0, 1, 3):
            assert board.gic.lines.get(TIMER_PPI, other) is False

    def test_cores_have_independent_timers(self):
        board = JetsonNanoA64Smp(ram_size=0x2000)
        timers = [c.registers.generic_timer for c in board.cores]
        assert len({id(t) for t in timers}) == 4
        timers[0].tick(100)
        assert timers[1].count == 0

    def test_single_core_board_still_drives_ppi_30(self):
        board = JetsonNanoA64(ram_size=0x2000)
        timer = board.cpu.registers.generic_timer
        timer.tval = 1
        timer.ctl = CTL_ENABLE
        board.sample_timer()
        assert board.gic.lines.get(TIMER_PPI, 0) is False
        timer.tick(10)
        board.sample_timer()
        assert board.gic.lines.get(TIMER_PPI, 0) is True


class TestBoardTimerFrequency:
    """CNTFRQ_EL0 is a board property, not an architectural constant."""

    @pytest.mark.parametrize('board_class,expected', [
        (RaspberryPi3A64, 19200000),
        (RaspberryPi4A64, 54000000),
        (JetsonNanoA64, 19200000),
    ])
    def test_cntfrq_matches_the_board(self, board_class, expected):
        board = board_class(ram_size=0x1000)
        assert board.cpu.registers.generic_timer.frequency == expected
        # And the firmware-visible register agrees with the model.
        assert board.cpu.registers.get_system_register(
            0b11, 0b011, 0b1110, 0b0000, 0b000) == expected

    def test_the_pi_4_is_not_the_pi_3(self):
        # 54 MHz vs 19.2 MHz. Firmware derives its tick period from CNTFRQ,
        # so confusing the two scales every delay by nearly three.
        assert RaspberryPi4.TIMER_FREQUENCY != RaspberryPi3.TIMER_FREQUENCY

    def test_every_core_in_a_cluster_gets_the_board_rate(self):
        board = JetsonNanoA64Smp(ram_size=0x2000)
        rates = {c.registers.generic_timer.frequency for c in board.cores}
        assert rates == {JetsonNanoA64Smp.TIMER_FREQUENCY}


class TestUnmappedAccessAborts:

    #: Nothing is mapped here on any board modelled below.
    NOWHERE = 0x0000_DEAD_0000

    def test_a64_boards_enable_the_check(self):
        assert JetsonNanoA64.FAULT_ON_UNMAPPED is True
        assert RaspberryPi4A64.FAULT_ON_UNMAPPED is True
        assert RaspberryPi3A64.FAULT_ON_UNMAPPED is True

    def test_armv6_boards_leave_it_off(self):
        # The ARMv6 model has no path that raises the external abort, and its
        # tests rely on unmapped reads returning zero.
        assert JetsonNano.FAULT_ON_UNMAPPED is False
        assert RaspberryPi4.FAULT_ON_UNMAPPED is False

    def test_hub_reports_mapped_and_unmapped(self):
        board = JetsonNanoA64(ram_size=0x2000)
        mem = board.cpu.mem
        assert mem.fault_on_unmapped is True
        assert mem.is_mapped(board.RAM_BASE) is True
        assert mem.is_mapped(self.NOWHERE) is False

    def test_an_access_straddling_the_end_of_a_region_is_unmapped(self):
        board = JetsonNanoA64(ram_size=0x2000)
        mem = board.cpu.mem
        last = board.RAM_BASE + 0x2000 - 4
        assert mem.is_mapped(last, 4) is True
        # The far half would silently read zero, which is the bug this
        # exists to catch, so the whole access counts as unmapped.
        assert mem.is_mapped(last, 8) is False

    def test_unmapped_store_raises_an_external_abort(self):
        from armulator.armv8.arm_exceptions import DataAbortException

        board = JetsonNanoA64(ram_size=0x2000)
        with pytest.raises(DataAbortException) as caught:
            board.cpu.translate_address(self.NOWHERE, is_write=True, size=4)
        # 0b010000 is "synchronous external abort, not on a table walk".
        assert caught.value.status == 0b010000
        assert caught.value.is_write is True

    def test_unmapped_fetch_raises_an_instruction_abort(self):
        from armulator.armv8.arm_exceptions import InstructionAbortException

        board = JetsonNanoA64(ram_size=0x2000)
        with pytest.raises(InstructionAbortException) as caught:
            board.cpu.translate_address(
                self.NOWHERE, size=4, is_instruction=True)
        assert caught.value.status == 0b010000

    def test_mapped_access_is_untouched(self):
        board = JetsonNanoA64(ram_size=0x2000)
        descriptor = board.cpu.translate_address(
            board.RAM_BASE + 0x100, is_write=True, size=4)
        assert descriptor.paddress.physicaladdress == board.RAM_BASE + 0x100

    def test_peripherals_are_mapped(self):
        board = JetsonNanoA64(ram_size=0x2000)
        for address in (board.UARTA_ADDRESS, board.GICD_ADDRESS):
            assert board.cpu.mem.is_mapped(address, 4) is True

    def test_the_check_can_be_turned_off(self):
        board = JetsonNanoA64(ram_size=0x2000)
        board.cpu.mem.fault_on_unmapped = False
        # No exception: back to the permissive read-as-zero behaviour.
        descriptor = board.cpu.translate_address(self.NOWHERE, size=4)
        assert descriptor is not None
