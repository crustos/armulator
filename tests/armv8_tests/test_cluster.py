"""
The multi-core cluster: shared state, PSCI bring-up, IPIs and the scheduler.
"""

import pytest
from keystone import KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN, Ks

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.armv6.memory_controller_hub import MemoryController
from armulator.armv8.cluster import Cluster
from armulator.armv8.enums import EL
from armulator.peripherals.gic400 import (
    GICC_CTLR,
    GICC_PMR,
    GICD_CTLR,
    GICD_ISENABLER,
    GICD_SGIR,
    Gic400,
)

RAM = [{'mem_type': 'RAM', 'beginning': 0x0, 'end': 0x200000}]
CODE = 0x1000
LOCK = 0x8000
COUNTER = 0x8100


@pytest.fixture(scope='module')
def assembler():
    return Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)


def load(cluster, assembler, address, source):
    code, _ = assembler.asm(source, address)
    for offset, byte in enumerate(bytes(code)):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address + offset
        cluster.memory[descriptor, 1] = byte


def read(cluster, address, size=4):
    descriptor = AddressDescriptor()
    descriptor.paddress.physicaladdress = address
    return cluster.memory[descriptor, size]


def write(cluster, address, value, size=4):
    descriptor = AddressDescriptor()
    descriptor.paddress.physicaladdress = address
    cluster.memory[descriptor, size] = value


class TestClusterConstruction:
    def test_cores_share_memory_and_the_monitor(self):
        cluster = Cluster(4, RAM)
        assert all(core.mem is cluster.memory for core in cluster.cores)
        assert all(core.exclusive_monitor is cluster.exclusive_monitor
                   for core in cluster.cores)

    def test_each_core_reports_its_own_mpidr(self):
        cluster = Cluster(4, RAM)
        for cpu_id, core in enumerate(cluster.cores):
            mpidr = core.registers.get_system_register(0b11, 0b000, 0b0000, 0b0000, 0b101)
            assert mpidr & 0xFF == cpu_id
            assert mpidr & 0x80000000     # bit 31 reads as one

    def test_only_the_primary_starts_running(self):
        cluster = Cluster(4, RAM)
        assert cluster.powered_on == [True, False, False, False]

    def test_a_write_by_one_core_is_seen_by_another(self):
        cluster = Cluster(2, RAM)
        cluster.cores[0].mem_set(0x9000, 8, 0xDEADBEEF)
        assert cluster.cores[1].mem_get(0x9000, 8) == 0xDEADBEEF


SPINLOCK = f'''
        movz x10, #{LOCK}
        movz x11, #{COUNTER}
        movz w12, #20
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
        stlr  wzr, [x10]
        sub   w12, w12, #1
        cbnz  w12, loop
        b .
'''

UNLOCKED = f'''
        movz x11, #{COUNTER}
        movz w12, #20
loop:   ldr   w4, [x11]
        add   w4, w4, #1
        str   w4, [x11]
        sub   w12, w12, #1
        cbnz  w12, loop
        b .
'''


class TestSpinlock:
    def _run(self, assembler, source, cores=4):
        cluster = Cluster(cores, RAM, slice_size=3)
        load(cluster, assembler, CODE, source)
        write(cluster, LOCK, 0)
        write(cluster, COUNTER, 0)
        for cpu_id in range(cores):
            cluster.power_on(cpu_id, CODE)
        cluster.run(500000)
        return cluster

    def test_lock_protects_a_shared_counter(self, assembler):
        cluster = self._run(assembler, SPINLOCK)
        # Four cores, twenty increments each, none lost.
        assert read(cluster, COUNTER) == 80
        assert read(cluster, LOCK) == 0

    def test_contention_actually_occurred(self, assembler):
        cluster = self._run(assembler, SPINLOCK)
        # If no exclusive store ever failed, the cores never actually raced and the
        # test above would prove nothing.
        assert cluster.exclusive_monitor.failures > 0
        assert cluster.exclusive_monitor.successes == 80

    def test_without_a_lock_increments_are_lost(self, assembler):
        cluster = self._run(assembler, UNLOCKED)
        assert read(cluster, COUNTER) < 80


class TestPsci:
    def test_cpu_on_releases_a_secondary(self, assembler):
        cluster = Cluster(2, RAM, slice_size=8)
        load(cluster, assembler, CODE, '''
                movz x0, #0x0003
                movk x0, #0xC400, lsl #16
                movz x1, #1
                movz x2, #0x3000
                movz x3, #0xABC
                smc  #0
                mov  x21, x0
                b .
        ''')
        load(cluster, assembler, 0x3000, 'movz x9, #0x55\n b .')
        cluster.power_on(0, CODE)
        cluster.run(5000)
        assert cluster.cores[0].registers.get_x(21) == 0       # PSCI_SUCCESS
        assert cluster.powered_on[1] is True
        assert cluster.cores[1].registers.get_x(9) == 0x55

    def test_context_id_arrives_in_x0(self, assembler):
        cluster = Cluster(2, RAM, slice_size=8)
        load(cluster, assembler, CODE, '''
                movz x0, #0x0003
                movk x0, #0xC400, lsl #16
                movz x1, #1
                movz x2, #0x3000
                movz x3, #0xABC
                smc  #0
                b .
        ''')
        load(cluster, assembler, 0x3000, 'mov x9, x0\n b .')
        cluster.power_on(0, CODE)
        cluster.run(5000)
        assert cluster.cores[1].registers.get_x(9) == 0xABC

    def test_starting_a_running_core_reports_already_on(self, assembler):
        cluster = Cluster(2, RAM, slice_size=8)
        load(cluster, assembler, CODE, '''
                movz x0, #0x0003
                movk x0, #0xC400, lsl #16
                movz x1, #1
                movz x2, #0x3000
                smc  #0
                movz x0, #0x0003
                movk x0, #0xC400, lsl #16
                movz x1, #1
                movz x2, #0x3000
                smc  #0
                mov  x22, x0
                b .
        ''')
        load(cluster, assembler, 0x3000, 'b .')
        cluster.power_on(0, CODE)
        cluster.run(5000)
        assert cluster.cores[0].registers.get_x(22) == 0xFFFFFFFFFFFFFFFC   # ALREADY_ON

    def test_affinity_info_reports_off_then_on(self, assembler):
        cluster = Cluster(2, RAM, slice_size=8)
        load(cluster, assembler, CODE, '''
                movz x0, #0x0004
                movk x0, #0xC400, lsl #16
                movz x1, #1
                smc  #0
                mov  x21, x0
                movz x0, #0x0003
                movk x0, #0xC400, lsl #16
                movz x1, #1
                movz x2, #0x3000
                smc  #0
                movz x0, #0x0004
                movk x0, #0xC400, lsl #16
                movz x1, #1
                smc  #0
                mov  x22, x0
                b .
        ''')
        load(cluster, assembler, 0x3000, 'b .')
        cluster.power_on(0, CODE)
        cluster.run(5000)
        assert cluster.cores[0].registers.get_x(21) == 1    # OFF
        assert cluster.cores[0].registers.get_x(22) == 0    # ON

    def test_invalid_target_is_rejected(self, assembler):
        cluster = Cluster(2, RAM, slice_size=8)
        load(cluster, assembler, CODE, '''
                movz x0, #0x0003
                movk x0, #0xC400, lsl #16
                movz x1, #9              // no such core
                movz x2, #0x3000
                smc  #0
                mov  x21, x0
                b .
        ''')
        cluster.power_on(0, CODE)
        cluster.run(5000)
        assert cluster.cores[0].registers.get_x(21) == 0xFFFFFFFFFFFFFFFE  # INVALID


class TestInterProcessorInterrupts:
    GIC_ADDRESS = 0x50041000
    VBAR = 0x4000
    MARKER = 0x9100

    def _cluster_with_gic(self, assembler):
        cluster = Cluster(2, RAM, slice_size=4)
        gic = Gic400(name='gic')
        cluster.gic = gic
        cluster.memory.memories.append(
            MemoryController(gic, self.GIC_ADDRESS, self.GIC_ADDRESS + gic.size)
        )
        gic.write_register(GICD_CTLR, 1)
        for cpu_id in range(2):
            gic.current_cpu = cpu_id
            gic.write_register(GICC_CTLR, 1)
            gic.write_register(GICC_PMR, 0xFF)
        gic.write_register(GICD_ISENABLER, 1 << 3)
        for core in cluster.cores:
            core.registers.vbar[EL.EL1] = self.VBAR
        return cluster, gic

    def test_sgi_wakes_a_core_from_wfi(self, assembler):
        cluster, gic = self._cluster_with_gic(assembler)
        load(cluster, assembler, 0x1000, 'msr daifclr, #2\n wfi\n b .')
        load(cluster, assembler, self.VBAR + 0x280,
             f'movz x5, #{self.MARKER}\n movz w6, #0xABC\n str w6, [x5]\n b .')
        load(cluster, assembler, 0x1800, f'''
                movz x0, #{self.GIC_ADDRESS & 0xFFFF}
                movk x0, #{self.GIC_ADDRESS >> 16}, lsl #16
                movz w1, #0x0003
                movk w1, #0x0002, lsl #16      // target cpu1, SGI 3
                str  w1, [x0, #{GICD_SGIR}]
                b .
        ''')
        write(cluster, self.MARKER, 0)
        cluster.power_on(1, 0x1000)
        cluster.run(200)
        assert cluster.cores[1].is_wait_for_interrupt is True

        cluster.power_on(0, 0x1800)
        cluster.run(5000)
        assert read(cluster, self.MARKER) == 0xABC

    def test_an_sgi_targeted_elsewhere_is_not_delivered(self, assembler):
        cluster, gic = self._cluster_with_gic(assembler)
        gic.send_sgi(3, target_cpus=0b10)          # cpu1 only
        assert gic.irq_pending_for(1) is True
        assert gic.irq_pending_for(0) is False

    def test_a_broadcast_sgi_reaches_every_target(self, assembler):
        cluster, gic = self._cluster_with_gic(assembler)
        gic.send_sgi(3, target_cpus=0b11)
        assert gic.irq_pending_for(0) is True
        assert gic.irq_pending_for(1) is True
        # One core acknowledging must not consume the other core's copy.
        gic.current_cpu = 0
        assert gic.acknowledge() == 3
        assert gic.irq_pending_for(0) is False
        assert gic.irq_pending_for(1) is True


class TestScheduling:
    def test_wfe_parks_and_sev_releases(self, assembler):
        cluster = Cluster(2, RAM, slice_size=4)
        load(cluster, assembler, 0x1000, 'wfe\n movz x9, #0x77\n b .')
        load(cluster, assembler, 0x1800, 'sev\n b .')
        cluster.power_on(1, 0x1000)
        cluster.run(100)
        assert cluster.cores[1].is_wait_for_event is True

        cluster.power_on(0, 0x1800)
        cluster.run(2000)
        assert cluster.cores[1].registers.get_x(9) == 0x77

    def test_wfe_returns_immediately_when_an_event_is_pending(self, assembler):
        # The event register absorbs a SEV that arrived before the WFE, so the core
        # must not park and wait for another one.
        cluster = Cluster(1, RAM, slice_size=4)
        load(cluster, assembler, 0x1000, 'sev\n wfe\n movz x9, #0x33\n b .')
        cluster.power_on(0, 0x1000)
        cluster.run(200)
        assert cluster.cores[0].registers.get_x(9) == 0x33

    def test_halt_detection_stops_the_run(self, assembler):
        cluster = Cluster(2, RAM, slice_size=4)
        load(cluster, assembler, 0x1000, 'movz x0, #1\n b .')
        cluster.power_on(0, 0x1000)
        cluster.power_on(1, 0x1000)
        executed = cluster.run(100000)
        # Both cores park quickly, so the budget must not be consumed.
        assert executed < 200
        assert cluster.all_halted is True

    def test_a_powered_off_core_does_not_execute(self, assembler):
        cluster = Cluster(2, RAM, slice_size=4)
        load(cluster, assembler, 0x1000, 'movz x9, #1\n b .')
        cluster.power_on(0, 0x1000)
        cluster.run(500)
        assert cluster.cores[1].registers.get_x(9) == 0
