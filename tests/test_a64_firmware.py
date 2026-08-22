"""
AArch64 firmware running against the real peripheral models.

These are the counterparts of the ARMv6 firmware tests in test_devices.py: the same
peripherals, the same register sequences, driven by A64 code on the AArch64 core.
"""

import pytest

from armulator.armv8.enums import EL
from armulator.boards import JetsonNanoA64, RaspberryPi4A64
from armulator.boards.firmware import HAVE_KEYSTONE, firmware_a64
from armulator.peripherals.gic400 import (
    GICC_CTLR,
    GICC_PMR,
    GICD_CTLR,
    GICD_ISENABLER,
    SPI_BASE,
)

pytestmark = pytest.mark.skipif(
    not HAVE_KEYSTONE, reason='keystone-engine required to assemble firmware'
)

VBAR = 0x80090000


def run(board, source, budget=2000):
    board.load(board.CODE_BASE, firmware_a64(source, address=board.CODE_BASE))
    board.start()
    board.run(budget)
    return board


class TestJetsonGpio:
    def test_firmware_drives_a_pin(self):
        board = run(JetsonNanoA64(), """
                movz x0, #0x6000, lsl #16
                movk x0, #0xD000
                movz w1, #1
                str  w1, [x0, #0x00]        // CNF port A -> GPIO
                str  w1, [x0, #0x10]        // OE  -> output
                str  w1, [x0, #0x20]        // OUT -> high
        """)
        assert board.gpio.level('PA0') is True
        assert board.halted is True

    def test_firmware_reads_an_input_pin(self):
        board = JetsonNanoA64()
        board.gpio.write_register(0x00, 1 << 1)      # CNF: PA1 as GPIO
        board.gpio.drive_input('PA1', True)
        run(board, """
                movz x0, #0x6000, lsl #16
                movk x0, #0xD000
                ldr  w1, [x0, #0x30]        // IN port A
                and  w2, w1, #2
        """)
        assert board.cpu.registers.get_x(2) == 2

    def test_firmware_toggles_a_pin_in_a_loop(self):
        # Exercises a backward branch, a counter and a read-modify-write.
        board = run(JetsonNanoA64(), """
                movz x0, #0x6000, lsl #16
                movk x0, #0xD000
                movz w1, #1
                str  w1, [x0, #0x00]
                str  w1, [x0, #0x10]
                movz w3, #4                 // toggle count
        loop:   ldr  w2, [x0, #0x20]
                eor  w2, w2, #1
                str  w2, [x0, #0x20]
                sub  w3, w3, #1
                cbnz w3, loop
        """)
        # An even number of toggles from low leaves the pin low again.
        assert board.gpio.level('PA0') is False
        assert board.halted is True


class TestPi4Uart:
    def test_firmware_prints_over_uart(self):
        board = run(RaspberryPi4A64(), """
                movz x0, #0xFE20, lsl #16
                movk x0, #0x1000
                movz w1, #0x4F
                str  w1, [x0]
                movz w1, #0x4B
                str  w1, [x0]
        """)
        assert board.uart.text == 'OK'

    def test_firmware_polls_the_flag_register(self):
        # A subroutine with BL/RET plus a TBNZ poll loop, as real driver code writes it.
        board = run(RaspberryPi4A64(trace=True), """
                movz x0, #0xFE20, lsl #16
                movk x0, #0x1000
                movz w2, #0x41
                bl   putc
                b    done
        putc:   ldr  w1, [x0, #0x18]        // FR
                tbnz w1, #5, putc           // TXFF still set, keep waiting
                str  w2, [x0]
                ret
        done:   nop
        """)
        assert board.uart.text == 'A'
        assert board.uart.reads_of('FR')


class TestMemoryFirmware:
    def test_firmware_fills_and_sums_a_buffer(self):
        board = run(JetsonNanoA64(), """
                movz x0, #0x800A, lsl #16
                movz w1, #0
                movz w2, #8
        fill:   str  w1, [x0, w1, uxtw #2]
                add  w1, w1, #1
                cmp  w1, w2
                b.lo fill
                movz w1, #0
                movz w3, #0
        sum:    ldr  w4, [x0, w1, uxtw #2]
                add  w3, w3, w4
                add  w1, w1, #1
                cmp  w1, w2
                b.lo sum
        """)
        assert board.cpu.registers.get_x(3) == 28      # 0 + 1 + ... + 7


class TestInterruptHandler:
    def _armed_board(self):
        board = JetsonNanoA64()
        board.cpu.registers.vbar[EL.EL1] = VBAR
        intid = SPI_BASE + board.GPIO_SPI
        board.gic.write_register(GICD_CTLR, 1)
        board.gic.write_register(GICC_CTLR, 1)
        board.gic.write_register(GICC_PMR, 0xFF)
        board.gic.write_register(GICD_ISENABLER + 4 * (intid // 32), 1 << (intid % 32))
        board.gpio.write_register(0x50, 1 << 0)        # INT_ENB on PA0
        board.load(board.CODE_BASE, firmware_a64('movz x20, #0', address=board.CODE_BASE))
        board.load(VBAR + 0x280, firmware_a64("""
                movz x20, #0xBEEF            // evidence the handler ran
                movz x0, #0x6000, lsl #16
                movk x0, #0xD000
                movz w1, #1
                str  w1, [x0, #0x70]         // INT_CLR - acknowledge at the source
                eret
        """, address=VBAR + 0x280))
        board.start()
        board.cpu.registers.pstate.i = 0
        board.run(4)
        return board

    def test_handler_runs_and_returns(self):
        board = self._armed_board()
        resume_pc = board.cpu_adapter.pc
        board.gpio.drive_input('PA0', True)
        assert board.service_interrupts() is True
        assert board.cpu_adapter.pc == VBAR + 0x280

        for _ in range(6):
            board.step()

        assert board.cpu.registers.get_x(20) == 0xBEEF
        assert board.cpu_adapter.pc == resume_pc
        # ERET restores PSTATE, so interrupts are unmasked again.
        assert board.cpu.registers.pstate.i == 0

    def test_unacknowledged_level_interrupt_refires(self):
        # A handler that does not clear the source is re-entered immediately, which is
        # what real level-triggered hardware does.
        board = self._armed_board()
        board.load(VBAR + 0x280, firmware_a64('movz x20, #1\n eret', address=VBAR + 0x280))
        board.gpio.drive_input('PA0', True)
        board.service_interrupts()
        for _ in range(6):
            board.step()
        # Execution never escapes the handler: each ERET is followed by another entry.
        assert VBAR + 0x280 <= board.cpu_adapter.pc < VBAR + 0x290
        assert board.cpu.registers.get_x(20) == 1
