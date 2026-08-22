"""
The Jetson Nano's console, interrupt controller base and architected timer.

Each of these was wrong in a way that produced no error: a PL011 standing in
for a 16550 accepts the first byte and then hangs the driver, a GIC base off
by 0x1000 reads back as zero rather than faulting, and a counter that never
advances turns a delay loop into a hang. So the assertions here are mostly
about *which* register was touched, not just about the end result.
"""

import pytest

from armulator.armv8.generic_timer import (
    CTL_ENABLE, CTL_IMASK, CTL_ISTATUS, TIMER_PPI, GenericTimer,
)
from armulator.boards import JetsonNano, JetsonNanoA64
from armulator.peripherals.gic400 import (
    GICC_CTLR, GICC_PMR, GICD_CTLR, GICD_ISENABLER,
)
from armulator.peripherals.uart_8250 import (
    LSR_DR, LSR_THRE, REG_FCR, REG_IER, REG_LCR, REG_LSR, REG_MCR, REG_THR,
    TegraUart, Uart8250,
)

#: UART-A's input clock, and the divisor 115200 baud implies.
TEGRA_UART_CLOCK = 408000000
TEGRA_115200_DIVISOR = 0xDD


def write(uart, index, value):
    """Write a register by 16550 index, honouring the device's spacing."""
    uart.write(index << uart.shift, 4, value)


def read(uart, index):
    return int.from_bytes(uart.read(index << uart.shift, 4), 'little')


def init_sequence(uart, clock=TEGRA_UART_CLOCK, baud=115200):
    """The initialisation an ordinary 16550 driver performs."""
    divisor = clock // (16 * baud)
    write(uart, REG_IER, 0)
    write(uart, REG_LCR, 0x80)                 # DLAB
    write(uart, 0, divisor & 0xFF)             # DLL
    write(uart, 1, (divisor >> 8) & 0xFF)      # DLM
    write(uart, REG_LCR, 0x03)                 # 8N1, DLAB clear
    write(uart, REG_FCR, 0x07)                 # enable + clear both FIFOs
    write(uart, REG_MCR, 0x03)                 # DTR | RTS
    return divisor


class TestTegraUartRegisters:

    def test_divisor_is_reachable_only_behind_dlab(self):
        uart = TegraUart()
        write(uart, REG_LCR, 0x80)
        write(uart, 0, 0xDD)
        assert uart.divisor == 0xDD
        # With DLAB clear the same offset is the transmit register, so a
        # character must go to the wire and leave the divisor alone.
        write(uart, REG_LCR, 0x03)
        write(uart, REG_THR, ord('A'))
        assert uart.divisor == 0xDD
        assert uart.tx_buffer == b'A'

    def test_init_sequence_programs_115200(self):
        uart = TegraUart()
        divisor = init_sequence(uart)
        assert divisor == TEGRA_115200_DIVISOR
        assert uart.divisor == TEGRA_115200_DIVISOR
        # DLAB must be left clear, or every later character lands in the
        # divisor latch and nothing is ever transmitted.
        assert uart.dlab is False
        assert uart.fifos_enabled is True
        assert uart.baud(TEGRA_UART_CLOCK) == pytest.approx(115200, rel=0.01)

    def test_registers_are_spaced_four_bytes_apart(self):
        uart = TegraUart()
        assert uart.shift == 2
        # LCR is register 3, so it lives at 0x0C and nowhere else. Writing
        # the unshifted offset 0x03 must not reach it -- that is the mistake
        # that silently configures the port through IIR/FCR.
        write(uart, REG_LCR, 0x80)
        assert uart.dlab is True
        uart2 = TegraUart()
        uart2.write(0x03, 1, 0x80)
        assert uart2.dlab is False

    def test_thre_is_set_so_a_polling_driver_makes_progress(self):
        uart = TegraUart()
        init_sequence(uart)
        assert read(uart, REG_LSR) & LSR_THRE
        # The polarity is the opposite of the PL011's TXFF: a driver waits
        # while THRE is *clear*, so a model that left it clear would hang.
        for ch in b'Jetson':
            assert read(uart, REG_LSR) & LSR_THRE
            write(uart, REG_THR, ch)
        assert uart.text == 'Jetson'

    def test_received_bytes_raise_dr_and_read_back(self):
        uart = TegraUart()
        init_sequence(uart)
        assert not read(uart, REG_LSR) & LSR_DR
        uart.feed('hi')
        assert read(uart, REG_LSR) & LSR_DR
        assert read(uart, 0) == ord('h')
        assert read(uart, 0) == ord('i')
        assert not read(uart, REG_LSR) & LSR_DR

    def test_rx_interrupt_follows_ier(self):
        uart = TegraUart()
        init_sequence(uart)
        uart.feed('x')
        assert uart.irq_pending is False       # ERBFI still clear
        write(uart, REG_IER, 0x01)
        assert uart.irq_pending is True
        read(uart, 0)                          # draining clears the source
        assert uart.irq_pending is False

    def test_unshifted_variant_places_registers_one_byte_apart(self):
        uart = Uart8250(shift=0)
        init_sequence(uart)
        assert uart.divisor == TEGRA_115200_DIVISOR
        assert uart.dlab is False


class TestJetsonBoardConsole:

    def test_console_is_a_16550_not_a_pl011(self):
        board = JetsonNano()
        assert isinstance(board.uart, Uart8250)

    def test_console_is_mapped_at_uart_a(self):
        board = JetsonNano()
        board.uart.write(0x00, 4, ord('J'))
        assert board.uart.text == 'J'
        assert board.UARTA_ADDRESS == 0x70006000

    def test_gic_distributor_and_cpu_interface_sit_where_tegra_puts_them(self):
        board = JetsonNano()
        # The datasheet and device tree quote 0x50041000 for the distributor,
        # which is the GIC base plus 0x1000 -- not the base itself.
        assert board.GICD_ADDRESS == 0x50041000
        assert board.GICC_ADDRESS == 0x50042000
        assert board.GIC_ADDRESS == board.GICD_ADDRESS - 0x1000

    def test_distributor_writes_reach_the_datasheet_address(self):
        # The regression this guards: with the base set to 0x50041000 the
        # distributor answered at 0x50042000 instead, and firmware writing to
        # the address the device tree names saw its writes go nowhere -- and
        # read back zero rather than faulting.
        board = JetsonNano()
        gic = board.gic
        offset = board.GICD_ADDRESS - board.GIC_ADDRESS
        assert offset == GICD_CTLR
        gic.write(offset, 4, 1)
        assert gic.read_register(GICD_CTLR) & 1

    def test_cpu_interface_offset_matches_gic400_layout(self):
        board = JetsonNano()
        assert board.GICC_ADDRESS - board.GIC_ADDRESS == GICC_CTLR
        # Tegra's distributor-to-CPU-interface gap is 0x1000, not the
        # 0x10000 the qemu virt machine uses.
        assert board.GICC_ADDRESS - board.GICD_ADDRESS == 0x1000

    def test_uart_interrupt_is_intid_68(self):
        board = JetsonNano()
        # Tegra X1 routes UART-A to SPI 36, which is interrupt ID 68.
        assert board.gic.connect(board.uart, board.UART_SPI) == 68


class TestGenericTimer:

    def test_counter_advances_and_istatus_latches(self):
        timer = GenericTimer(frequency=19200000, ticks_per_instruction=1)
        timer.tval = 100
        timer.ctl = CTL_ENABLE
        assert timer.istatus is False
        for _ in range(100):
            timer.tick()
        assert timer.istatus is True
        assert timer.ctl & CTL_ISTATUS

    def test_imask_hides_the_interrupt_but_not_the_condition(self):
        timer = GenericTimer(ticks_per_instruction=1)
        timer.tval = 5
        timer.ctl = CTL_ENABLE | CTL_IMASK
        for _ in range(10):
            timer.tick()
        assert timer.istatus is True           # the condition still holds
        assert timer.irq_pending is False      # but the line stays low
        timer.ctl = CTL_ENABLE
        assert timer.irq_pending is True

    def test_istatus_is_not_writable(self):
        timer = GenericTimer(ticks_per_instruction=1)
        timer.tval = 500                       # a compare well ahead of now
        timer.ctl = CTL_ENABLE | CTL_ISTATUS
        # ISTATUS is recomputed from the counter, never stored, so writing it
        # set must not make it read back set.
        assert timer.istatus is False
        assert not timer.ctl & CTL_ISTATUS

    def test_istatus_holds_at_reset_when_cval_is_zero(self):
        # Not a quirk of the model: with CVAL at its reset value of 0 the
        # counter is already >= it, so an enabled timer asserts immediately.
        # This is why firmware programs TVAL before setting ENABLE.
        timer = GenericTimer(ticks_per_instruction=1)
        timer.ctl = CTL_ENABLE
        assert timer.istatus is True

    def test_tval_is_a_delta_from_now(self):
        timer = GenericTimer(ticks_per_instruction=1)
        timer.tick(1000)
        timer.tval = 50
        assert timer.compare == 1050
        timer.tick(20)
        assert timer.tval == 30

    def test_disabled_timer_never_asserts(self):
        timer = GenericTimer(ticks_per_instruction=1)
        timer.tval = 1
        for _ in range(10):
            timer.tick()
        assert timer.istatus is False
        assert timer.irq_pending is False


class TestTimerRegisterEncodings:
    """MRS/MRS encodings, so a mis-decoded register is caught here."""

    CNTFRQ = (0b11, 0b011, 0b1110, 0b0000, 0b000)
    CNTPCT = (0b11, 0b011, 0b1110, 0b0000, 0b001)
    CNTP_TVAL = (0b11, 0b011, 0b1110, 0b0010, 0b000)
    CNTP_CTL = (0b11, 0b011, 0b1110, 0b0010, 0b001)
    CNTP_CVAL = (0b11, 0b011, 0b1110, 0b0010, 0b010)

    def registers(self):
        return JetsonNanoA64(ram_size=0x1000).cpu.registers

    def test_cntfrq_reports_the_jetsons_19_2mhz(self):
        assert self.registers().get_system_register(*self.CNTFRQ) == 19200000

    def test_cntpct_advances(self):
        regs = self.registers()
        before = regs.get_system_register(*self.CNTPCT)
        regs.generic_timer.tick(10)
        assert regs.get_system_register(*self.CNTPCT) == before + 10

    def test_cntpct_is_read_only(self):
        regs = self.registers()
        regs.set_system_register(*self.CNTPCT, 12345)
        assert regs.get_system_register(*self.CNTPCT) == 0

    def test_writing_tval_sets_cval(self):
        regs = self.registers()
        regs.generic_timer.tick(100)
        regs.set_system_register(*self.CNTP_TVAL, 500)
        assert regs.get_system_register(*self.CNTP_CVAL) == 600

    def test_ctl_round_trips_and_reports_istatus(self):
        regs = self.registers()
        regs.set_system_register(*self.CNTP_TVAL, 10)
        regs.set_system_register(*self.CNTP_CTL, CTL_ENABLE)
        assert regs.get_system_register(*self.CNTP_CTL) == CTL_ENABLE
        regs.generic_timer.tick(10)
        assert regs.get_system_register(*self.CNTP_CTL) & CTL_ISTATUS


class TestTimerReachesTheGic:

    def test_timer_ppi_is_30(self):
        assert TIMER_PPI == 30

    def test_board_samples_the_timer_into_the_distributor(self):
        board = JetsonNanoA64(ram_size=0x1000)
        timer = board.cpu.registers.generic_timer
        timer.tval = 5
        timer.ctl = CTL_ENABLE
        board.sample_timer()
        assert board.gic.lines[TIMER_PPI] is False
        timer.tick(10)
        board.sample_timer()
        assert board.gic.lines[TIMER_PPI] is True

    def test_enabling_the_ppi_makes_the_gic_assert(self):
        board = JetsonNanoA64(ram_size=0x1000)
        gic = board.gic
        gic.write(GICD_CTLR, 4, 1)
        gic.write(GICC_CTLR, 4, 1)
        gic.write(GICC_PMR, 4, 0xF0)
        gic.write(GICD_ISENABLER + (TIMER_PPI // 32) * 4, 4,
                  1 << (TIMER_PPI % 32))

        timer = board.cpu.registers.generic_timer
        timer.tval = 1
        timer.ctl = CTL_ENABLE
        timer.tick(10)
        board.sample_timer()
        gic.refresh()
        assert gic.irq_pending is True
