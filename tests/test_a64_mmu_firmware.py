"""
AArch64 firmware running with the MMU enabled against the real peripheral models.

This is the shape a real kernel takes: peripherals reached through a virtual mapping
typed as Device memory, RAM identity-mapped so the code keeps running across the moment
the MMU comes on.
"""

import pytest

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.armv6.memory_attributes import MemType
from armulator.boards import JetsonNanoA64
from armulator.boards.firmware import HAVE_KEYSTONE, firmware_a64

pytestmark = pytest.mark.skipif(
    not HAVE_KEYSTONE, reason='keystone-engine required to assemble firmware'
)

VALID = 1
TABLE = 2
ACCESS_FLAG = 1 << 10
INNER_SHAREABLE = 3 << 8
AP_EL1_RW = 0b00 << 6
ATTR_NORMAL = 0 << 2
ATTR_DEVICE = 1 << 2

#: Where the firmware expects to find the GPIO block once translation is on.
GPIO_VA = 0xC0000000
PAGE_TABLE_BASE = 0x80200000


def build_tables(board):
    """
    Identity-map RAM and the peripheral region with 1GB blocks, then map the GPIO
    controller again at GPIO_VA as Device memory.
    """
    allocated = [PAGE_TABLE_BASE]

    def write64(address, value):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address
        board.cpu.mem[descriptor, 8] = value

    def allocate():
        table = allocated[0]
        allocated[0] += 0x1000
        for offset in range(0, 0x1000, 8):
            write64(table + offset, 0)
        return table

    root = allocate()
    level1 = allocate()
    write64(root, level1 | TABLE | VALID)

    block = VALID | ACCESS_FLAG | INNER_SHAREABLE | AP_EL1_RW
    # VA 1GB -> PA 1GB covers the Tegra peripheral window; VA 2GB -> PA 2GB covers RAM.
    write64(level1 + 1 * 8, 0x40000000 | block | ATTR_DEVICE)
    write64(level1 + 2 * 8, 0x80000000 | block | ATTR_NORMAL)

    # A 4KB page putting the GPIO controller at GPIO_VA.
    level2 = allocate()
    level3 = allocate()
    write64(level1 + ((GPIO_VA >> 30) & 0x1FF) * 8, level2 | TABLE | VALID)
    write64(level2 + ((GPIO_VA >> 21) & 0x1FF) * 8, level3 | TABLE | VALID)
    write64(level3 + ((GPIO_VA >> 12) & 0x1FF) * 8,
            board.GPIO_ADDRESS | TABLE | VALID | ACCESS_FLAG | INNER_SHAREABLE
            | AP_EL1_RW | ATTR_DEVICE)

    registers = board.cpu.registers
    registers.set_system_register(0b11, 0b000, 0b1010, 0b0010, 0b000, 0xFF | (0x00 << 8))
    tcr = 16 | (16 << 16) | (0b00 << 14) | (0b10 << 30)
    registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b010, tcr)
    registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b000, root)
    registers.set_system_register(0b11, 0b000, 0b0010, 0b0000, 0b001, root)
    return root


ENABLE_MMU = '''
        mrs  x3, sctlr_el1
        orr  x3, x3, #1
        msr  sctlr_el1, x3
        isb
'''


def run(board, body, budget=500):
    board.load(board.CODE_BASE, firmware_a64(ENABLE_MMU + body, address=board.CODE_BASE))
    board.start()
    board.run(budget)
    return board


@pytest.fixture
def board():
    machine = JetsonNanoA64(ram_size=0x400000)
    build_tables(machine)
    return machine


class TestFirmwareWithTranslation:
    def test_gpio_driven_through_a_virtual_address(self, board):
        run(board, '''
                movz x0, #0xC000, lsl #16
                movz w1, #1
                str  w1, [x0, #0x00]
                str  w1, [x0, #0x10]
                str  w1, [x0, #0x20]
        ''')
        assert board.cpu.registers.mmu_enabled is True
        assert board.gpio.level('PA0') is True
        assert board.halted is True

    def test_firmware_keeps_running_across_enabling_the_mmu(self, board):
        # Code is identity-mapped, so the instruction after the enabling ISB fetches
        # from the same address it would have without translation.
        board_after = run(board, 'movz x9, #0x1234')
        assert board_after.cpu.registers.get_x(9) == 0x1234
        assert board_after.fault_loop is False

    def test_peripheral_mapping_is_device_memory(self, board):
        run(board, 'nop')
        descriptor = board.cpu.translate_address(GPIO_VA)
        assert descriptor.paddress.physicaladdress == board.GPIO_ADDRESS
        assert descriptor.memattrs.type == MemType.DEVICE

    def test_ram_is_identity_mapped_as_normal_memory(self, board):
        run(board, 'nop')
        descriptor = board.cpu.translate_address(board.CODE_BASE)
        assert descriptor.paddress.physicaladdress == board.CODE_BASE
        assert descriptor.memattrs.type == MemType.NORMAL

    def test_read_modify_write_of_a_peripheral_register(self, board):
        run(board, '''
                movz x0, #0xC000, lsl #16
                movz w1, #1
                str  w1, [x0, #0x00]
                str  w1, [x0, #0x10]
                str  w1, [x0, #0x20]
                ldr  w2, [x0, #0x20]
                eor  w2, w2, #1
                str  w2, [x0, #0x20]
        ''')
        assert board.gpio.level('PA0') is False

    def test_unmapped_access_faults_rather_than_reaching_memory(self, board):
        # 0x10000000 lies in the first gigabyte, which these tables never map. Without
        # translation the store would simply fall off the memory map and be discarded;
        # with it, the access must fault.
        from armulator.armv8.enums import EL
        vbar = 0x80090000
        board.cpu.registers.vbar[EL.EL1] = vbar
        # The synchronous handler has to park, otherwise it runs off into unwritten
        # memory, faults again and overwrites the syndrome we came to inspect.
        board.load(vbar + 0x200, firmware_a64('', address=vbar + 0x200))
        run(board, '''
                movz x0, #0x1000, lsl #16
                movz w1, #1
                str  w1, [x0]
        ''')
        esr = board.cpu.registers.esr[EL.EL1]
        assert esr >> 26 == 0b100101              # data abort from the current EL
        assert (esr >> 6) & 1 == 1                # WnR: it was a write
        assert (esr & 0b111111) >> 2 == 0b0001    # translation fault
        assert board.cpu.registers.far[EL.EL1] == 0x10000000
