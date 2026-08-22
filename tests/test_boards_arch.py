"""
Board layer across both architectures.

The board layer talks to its processor through a CPU adapter, so these tests check that
the same board wiring works with either core and that the AArch64 specific paths -
interrupt delivery, flat addressing, 64-bit MMIO - behave.
"""

import pytest

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.boards import (
    JetsonNano,
    JetsonNanoA64,
    RaspberryPi4,
    RaspberryPi4A64,
)
from armulator.boards.cpu import ArmV6Adapter, ArmV8Adapter, make_adapter
from armulator.boards.firmware import assemble_a64, firmware_a64
from armulator.armv8.enums import EL
from armulator.peripherals.gic400 import (
    GICC_CTLR,
    GICC_PMR,
    GICD_CTLR,
    GICD_ISENABLER,
    SPI_BASE,
)

VBAR = 0x80090000


def enable_gic(gic, intid):
    gic.write_register(GICD_CTLR, 1)
    gic.write_register(GICC_CTLR, 1)
    gic.write_register(GICC_PMR, 0xFF)
    gic.write_register(GICD_ISENABLER + 4 * (intid // 32), 1 << (intid % 32))


def write_physical(board, address, size, value):
    descriptor = AddressDescriptor()
    descriptor.paddress.physicaladdress = address
    board.cpu.mem[descriptor, size] = value


def read_physical(board, address, size):
    descriptor = AddressDescriptor()
    descriptor.paddress.physicaladdress = address
    return board.cpu.mem[descriptor, size]


class TestAdapterSelection:
    def test_make_adapter_by_name(self):
        assert isinstance(make_adapter('armv6'), ArmV6Adapter)
        assert isinstance(make_adapter('armv8'), ArmV8Adapter)

    def test_unknown_architecture_is_rejected(self):
        with pytest.raises(ValueError, match='unknown architecture'):
            make_adapter('armv9')

    def test_make_adapter_passes_through_instances(self):
        adapter = ArmV8Adapter()
        assert make_adapter(adapter) is adapter

    def test_boards_report_their_architecture(self):
        assert JetsonNano().arch == 'armv6'
        assert JetsonNanoA64().arch == 'armv8'

    def test_arch_can_be_overridden_per_instance(self):
        assert JetsonNano(arch='armv8').arch == 'armv8'


class TestA64BoardConstruction:
    def test_peripheral_map_matches_the_armv6_board(self):
        # Only the processor differs; the memory map must be identical.
        a32 = {name: (mc.beginning, mc.end)
               for name, mc in zip(['ram'] + list(JetsonNano().devices),
                                   JetsonNano().cpu.mem.memories)}
        a64 = {name: (mc.beginning, mc.end)
               for name, mc in zip(['ram'] + list(JetsonNanoA64().devices),
                                   JetsonNanoA64().cpu.mem.memories)}
        assert a32 == a64

    def test_mmu_is_off_so_addressing_is_flat(self):
        board = JetsonNanoA64()
        assert board.cpu.registers.mmu_enabled is False

    def test_devices_are_attached(self):
        board = JetsonNanoA64()
        assert set(board.devices) == {'gpio', 'uart', 'spi', 'gic'}

    def test_pi4_a64_also_builds(self):
        board = RaspberryPi4A64()
        assert board.arch == 'armv8'
        assert set(board.devices) == set(RaspberryPi4().devices)

    def test_thumb_is_rejected_on_aarch64(self):
        board = JetsonNanoA64()
        with pytest.raises(ValueError, match='no Thumb'):
            board.start(thumb=True)


class TestA64MemoryMappedIo:
    def test_32_bit_peripheral_access(self):
        board = JetsonNanoA64()
        write_physical(board, board.GPIO_ADDRESS, 4, 0xFF)
        assert board.gpio.read_register(0x00) == 0xFF

    def test_64_bit_access_spans_two_registers(self):
        # AArch64 can issue accesses the ARMv6 core never could, so a 64-bit store
        # must land as two consecutive 32-bit register writes.
        board = JetsonNanoA64()
        write_physical(board, board.GPIO_ADDRESS, 8, 0x0000000200000001)
        assert board.gpio.read_register(0x00) == 0x01
        assert board.gpio.read_register(0x04) == 0x02
        assert read_physical(board, board.GPIO_ADDRESS, 8) == 0x0000000200000001

    def test_ram_is_mapped_at_the_jetson_base(self):
        board = JetsonNanoA64()
        write_physical(board, board.RAM_BASE, 8, 0xDEADBEEFCAFEBABE)
        assert read_physical(board, board.RAM_BASE, 8) == 0xDEADBEEFCAFEBABE


class TestA64InterruptDelivery:
    def _armed_board(self):
        board = JetsonNanoA64()
        board.cpu.registers.vbar[EL.EL1] = VBAR
        board.start()
        board.cpu.registers.pstate.i = 0
        enable_gic(board.gic, SPI_BASE + board.GPIO_SPI)
        board.gpio.write_register(0x50, 1 << 0)
        return board

    def test_masked_interrupts_are_not_delivered(self):
        board = self._armed_board()
        board.cpu.registers.pstate.i = 1
        board.gpio.drive_input('PA0', True)
        assert board.service_interrupts() is False

    def test_irq_vectors_to_the_irq_slot(self):
        board = self._armed_board()
        board.gpio.drive_input('PA0', True)
        assert board.service_interrupts() is True
        # Same EL using SP_ELx puts the IRQ entry at VBAR + 0x280.
        assert board.cpu_adapter.pc == VBAR + 0x280

    def test_irq_saves_return_state_and_masks(self):
        board = self._armed_board()
        board.gpio.drive_input('PA0', True)
        board.service_interrupts()
        assert board.cpu.registers.elr[EL.EL1] == board.CODE_BASE
        # Interrupts must be masked on entry so the handler is not re-entered.
        assert board.cpu.registers.pstate.i == 1

    def test_gic_acknowledge_returns_the_gpio_interrupt(self):
        board = self._armed_board()
        board.gpio.drive_input('PA0', True)
        board.service_interrupts()
        assert board.gic.acknowledge() == SPI_BASE + board.GPIO_SPI

    def test_no_interrupt_means_no_delivery(self):
        board = self._armed_board()
        assert board.service_interrupts() is False


class TestA64Execution:
    def test_data_processing_firmware_runs(self):
        board = JetsonNanoA64()
        code = assemble_a64(
            'movz x0, #0x1234\n movk x0, #0xABCD, lsl #16\n add x1, x0, #1',
            board.CODE_BASE,
        )
        board.load(board.CODE_BASE, code)
        board.start()
        for _ in range(3):
            board.step()
        assert board.cpu.registers.get_x(0) == 0xABCD1234
        assert board.cpu.registers.get_x(1) == 0xABCD1235

    def test_halt_loop_is_detected(self):
        # `b .` is a real branch now, so firmware parks on it cleanly.
        board = JetsonNanoA64()
        board.cpu.registers.vbar[EL.EL1] = VBAR
        board.load(board.CODE_BASE, firmware_a64('movz x0, #1', board.CODE_BASE))
        board.start()
        board.run(20)
        assert board.halted is True
        assert board.fault_loop is False
        assert board.cpu.registers.get_x(0) == 1

    def test_fault_loop_is_not_reported_as_halted(self):
        # An undefined encoding vectors to the (empty) handler, which is itself
        # undefined, so the PC repeats exactly as it would on a halt loop. The board
        # must not mistake a crashing firmware for a finished one.
        board = JetsonNanoA64()
        board.cpu.registers.vbar[EL.EL1] = VBAR
        board.load(board.CODE_BASE, b'\x00\x00\x00\x00' * 4)   # unallocated encoding
        board.start()
        board.run(20)
        assert board.fault_loop is True
        assert board.halted is False

    def test_gpio_firmware_drives_a_pin(self):
        # The AArch64 counterpart of the ARMv6 Jetson firmware test: the same
        # register sequence against the same peripheral model.
        board = JetsonNanoA64(trace=True)
        board.load(board.CODE_BASE, firmware_a64("""
                movz x0, #0x6000, lsl #16
                movk x0, #0xD000
                movz w1, #1
                str  w1, [x0, #0x00]
                str  w1, [x0, #0x10]
                str  w1, [x0, #0x20]
        """, board.CODE_BASE))
        board.start()
        board.run(200)
        assert board.gpio.level('PA0') is True
        assert board.halted is True
        assert [a.name for a in board.gpio.accesses if a.kind == 'w'] == [
            'CNF_PA', 'OE_PA', 'OUT_PA'
        ]


class TestArmV6BoardsUnchanged:
    def test_a32_halt_loop_still_detected(self):
        from armulator.boards.firmware import firmware
        board = JetsonNano()
        board.load(board.CODE_BASE, firmware('mov r0, #1', address=board.CODE_BASE))
        board.start()
        board.run(20)
        assert board.halted is True
        assert board.fault_loop is False
