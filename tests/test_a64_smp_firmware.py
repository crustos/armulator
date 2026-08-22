"""
Quad-core Jetson firmware.

Core 0 sets up the GPIO block and releases the other three through PSCI; the secondaries
then contend for a lock while driving the same peripheral, which is where a broken
exclusive monitor would show up as a wrong toggle count.
"""

import pytest

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.boards import JetsonNanoA64Smp
from armulator.boards.firmware import HAVE_KEYSTONE, assemble_a64

pytestmark = pytest.mark.skipif(
    not HAVE_KEYSTONE, reason='keystone-engine required to assemble firmware'
)

LOCK = 0x80090000
COUNTER = 0x80090100
SECONDARY = 0x80085000
ITERATIONS = 10


def load(board, address, source):
    for offset, byte in enumerate(assemble_a64(source, address=address)):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address + offset
        board.cluster.memory[descriptor, 1] = byte


def read(board, address, size=4):
    descriptor = AddressDescriptor()
    descriptor.paddress.physicaladdress = address
    return board.cluster.memory[descriptor, size]


PRIMARY_SOURCE = f'''
        movz x0, #0x6000, lsl #16
        movk x0, #0xD000
        movz w1, #1
        str  w1, [x0, #0x00]           // CNF port A -> GPIO
        str  w1, [x0, #0x10]           // OE  -> output
        movz x20, #1
next:   movz x0, #0x0003
        movk x0, #0xC400, lsl #16      // PSCI CPU_ON
        mov  x1, x20
        movz x2, #{SECONDARY & 0xFFFF}
        movk x2, #{SECONDARY >> 16}, lsl #16
        mov  x3, x20
        smc  #0
        add  x20, x20, #1
        cmp  x20, #4
        b.lo next
        b .
'''

SECONDARY_SOURCE = f'''
        movz x10, #{LOCK & 0xFFFF}
        movk x10, #{LOCK >> 16}, lsl #16
        movz x11, #{COUNTER & 0xFFFF}
        movk x11, #{COUNTER >> 16}, lsl #16
        movz x12, #0x6000, lsl #16
        movk x12, #0xD000
        movz w13, #{ITERATIONS}
loop:
acquire:
        ldaxr w1, [x10]
        cbnz  w1, acquire
        movz  w2, #1
        stlxr w3, w2, [x10]
        cbnz  w3, acquire
        ldr   w4, [x11]
        add   w4, w4, #1
        str   w4, [x11]
        ldr   w5, [x12, #0x20]
        eor   w5, w5, #1
        str   w5, [x12, #0x20]
        stlr  wzr, [x10]
        sub   w13, w13, #1
        cbnz  w13, loop
        b .
'''


@pytest.fixture
def board():
    machine = JetsonNanoA64Smp(trace=True)
    machine.cluster.slice_size = 5
    load(machine, machine.CODE_BASE, PRIMARY_SOURCE)
    load(machine, SECONDARY, SECONDARY_SOURCE)
    machine.start()
    machine.run(200000)
    return machine


class TestQuadCoreJetson:
    def test_board_has_four_cores(self):
        machine = JetsonNanoA64Smp()
        assert len(machine.cores) == 4
        assert machine.arch == 'armv8'

    def test_every_core_sees_the_peripherals(self):
        machine = JetsonNanoA64Smp()
        for core in machine.cores:
            assert core.mem.get_memory_by_address(machine.GPIO_ADDRESS) is not None

    def test_secondaries_are_released_by_psci(self, board):
        assert board.cluster.powered_on == [True, True, True, True]

    def test_all_cores_finish(self, board):
        assert board.cluster.all_halted is True

    def test_lock_protected_counter_is_exact(self, board):
        # Three secondaries, ten iterations each, none lost to a race.
        assert read(board, COUNTER) == 3 * ITERATIONS
        assert read(board, LOCK) == 0

    def test_the_cores_actually_contended(self, board):
        monitor = board.cluster.exclusive_monitor
        assert monitor.successes == 3 * ITERATIONS
        assert monitor.failures > 0

    def test_the_peripheral_saw_every_toggle(self, board):
        toggles = [a for a in board.gpio.accesses
                   if a.kind == 'w' and a.name == 'OUT_PA']
        assert len(toggles) == 3 * ITERATIONS
        # An even number of toggles from low leaves the pin low.
        assert board.gpio.level('PA0') is False

    def test_single_core_board_is_unaffected(self):
        from armulator.boards import JetsonNanoA64
        machine = JetsonNanoA64()
        assert machine.cluster is None
        assert len(machine.cores) == 1
