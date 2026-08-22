"""
Relaxed memory ordering.

The tests here are litmus tests: tiny two-core programs whose outcome distinguishes one
memory model from another. Each is checked three ways - that sequential consistency
forbids the weak outcome, that a relaxed model produces it, and that the right barrier
puts it back. A test that only checked the last of those would pass against a model that
did nothing at all.
"""

import pytest
from keystone import KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN, Ks

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.armv8.cluster import Cluster
from armulator.armv8.store_buffer import MemoryModel, StoreBuffer

RAM = [{'mem_type': 'RAM', 'beginning': 0x0, 'end': 0x200000}]
# Far enough apart to land in different reservation granules.
X = 0x8000
Y = 0x8040
DATA = 0x8000
FLAG = 0x8040


@pytest.fixture(scope='module')
def assembler():
    return Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)


def load(cluster, assembler, address, source):
    code, _ = assembler.asm(source, address)
    for offset, byte in enumerate(bytes(code)):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address + offset
        cluster.memory[descriptor, 1] = byte


def write(cluster, address, value, size=4):
    descriptor = AddressDescriptor()
    descriptor.paddress.physicaladdress = address
    cluster.memory[descriptor, size] = value


def read(cluster, address, size=4):
    descriptor = AddressDescriptor()
    descriptor.paddress.physicaladdress = address
    return cluster.memory[descriptor, size]


class TestStoreBufferUnit:
    def test_sequential_model_does_not_buffer(self):
        buffer = StoreBuffer(model=MemoryModel.SEQUENTIAL)
        assert buffer.push(0x1000, 4, 1) is False
        assert len(buffer) == 0

    def test_relaxed_model_buffers(self):
        buffer = StoreBuffer(model=MemoryModel.RELAXED)
        assert buffer.push(0x1000, 4, 1) is True
        assert len(buffer) == 1

    def test_forwarding_returns_the_newest_byte(self):
        buffer = StoreBuffer(model=MemoryModel.RELAXED)
        buffer.push(0x1000, 4, 0x11223344)
        buffer.push(0x1000, 1, 0xFF)
        assert buffer.forward(0x1000, 4)[0x1000] == 0xFF

    def test_forwarding_covers_only_the_bytes_it_holds(self):
        buffer = StoreBuffer(model=MemoryModel.RELAXED)
        buffer.push(0x1001, 1, 0xAB)
        available = buffer.forward(0x1000, 4)
        assert available == {0x1001: 0xAB}

    def test_relaxed_drains_in_program_order(self):
        buffer = StoreBuffer(model=MemoryModel.RELAXED)
        buffer.push(0x1000, 4, 1)
        buffer.push(0x2000, 4, 2)
        written = []
        buffer.drain(lambda a, s, v: written.append(a))
        assert written == [0x1000, 0x2000]

    def test_adversarial_drains_in_reverse_order(self):
        buffer = StoreBuffer(model=MemoryModel.ADVERSARIAL)
        buffer.push(0x1000, 4, 1)
        buffer.push(0x2000, 4, 2)
        written = []
        buffer.drain(lambda a, s, v: written.append(a))
        assert written == [0x2000, 0x1000]


class TestStoreForwarding:
    def test_a_core_always_sees_its_own_writes(self, assembler):
        # Buffering must be invisible to the core doing the buffering, or ordinary
        # single-threaded code would break.
        cluster = Cluster(1, RAM, slice_size=1)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        load(cluster, assembler, 0x1000,
             f'movz x10, #{X}\n movz w1, #0x42\n str w1, [x10]\n ldr w2, [x10]\n b .')
        write(cluster, X, 0)
        cluster.power_on(0, 0x1000)
        cluster.run(500)
        assert cluster.cores[0].registers.get_x(2) == 0x42

    def test_partial_forwarding_merges_with_memory(self, assembler):
        cluster = Cluster(1, RAM, slice_size=1)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        load(cluster, assembler, 0x1000,
             f'movz x10, #{X}\n movz w1, #0xAB\n strb w1, [x10]\n ldr w2, [x10]\n b .')
        write(cluster, X, 0x11223344)
        cluster.power_on(0, 0x1000)
        cluster.run(500)
        # The buffered byte overlays the word already in memory.
        assert cluster.cores[0].registers.get_x(2) == 0x112233AB


def run_store_buffer_litmus(cluster, assembler, barrier=''):
    """
    Both cores write one location and then read the other. Sequential consistency
    forbids both reads returning zero; every real weak-memory machine allows it.
    """
    load(cluster, assembler, 0x1000,
         f'movz x10, #{X}\n movz x11, #{Y}\n movz w1, #1\n str w1, [x10]\n'
         f' {barrier}\n ldr w2, [x11]\n b .')
    load(cluster, assembler, 0x2000,
         f'movz x10, #{X}\n movz x11, #{Y}\n movz w1, #1\n str w1, [x11]\n'
         f' {barrier}\n ldr w2, [x10]\n b .')
    write(cluster, X, 0)
    write(cluster, Y, 0)
    cluster.power_on(0, 0x1000)
    cluster.power_on(1, 0x2000)
    cluster.run(20000)
    return cluster.cores[0].registers.get_x(2), cluster.cores[1].registers.get_x(2)


class TestStoreThenLoadReordering:
    def _cluster(self, model):
        cluster = Cluster(2, RAM, slice_size=1)
        cluster.set_memory_model(model)
        return cluster

    def test_sequential_consistency_forbids_both_zero(self, assembler):
        result = run_store_buffer_litmus(self._cluster(MemoryModel.SEQUENTIAL), assembler)
        assert result != (0, 0)

    def test_relaxed_model_allows_both_zero(self, assembler):
        # The outcome that breaks Dekker and Peterson written without barriers.
        result = run_store_buffer_litmus(self._cluster(MemoryModel.RELAXED), assembler)
        assert result == (0, 0)

    def test_a_barrier_restores_the_ordering(self, assembler):
        result = run_store_buffer_litmus(
            self._cluster(MemoryModel.ADVERSARIAL), assembler, barrier='dmb sy'
        )
        assert result != (0, 0)


def run_message_passing(cluster, assembler, writer_body, reader_body):
    """
    One core publishes data then sets a flag; the other spins on the flag and reads the
    data. Seeing the flag set but the data stale is the failure barriers prevent.
    """
    load(cluster, assembler, 0x1000,
         f'movz x10, #{DATA}\n movz x11, #{FLAG}\n' + writer_body + '\n b .')
    load(cluster, assembler, 0x2000,
         f'movz x10, #{DATA}\n movz x11, #{FLAG}\n' + reader_body + '\n b .')
    write(cluster, DATA, 0)
    write(cluster, FLAG, 0)
    cluster.power_on(1, 0x2000)      # the reader is already spinning
    cluster.power_on(0, 0x1000)
    cluster.run(40000)
    return cluster.cores[1].registers.get_x(3)


PLAIN_WRITER = 'movz w1, #42\n str w1, [x10]\n movz w2, #1\n str w2, [x11]'
BARRIER_WRITER = 'movz w1, #42\n str w1, [x10]\n dmb sy\n movz w2, #1\n str w2, [x11]'
RELEASE_WRITER = 'movz w1, #42\n str w1, [x10]\n movz w2, #1\n stlr w2, [x11]'
PLAIN_READER = 'spin: ldr w4, [x11]\n cbz w4, spin\n ldr w3, [x10]'
BARRIER_READER = 'spin: ldr w4, [x11]\n cbz w4, spin\n dmb sy\n ldr w3, [x10]'
ACQUIRE_READER = 'spin: ldar w4, [x11]\n cbz w4, spin\n ldr w3, [x10]'


class TestStoreThenStoreReordering:
    def _cluster(self, model):
        cluster = Cluster(2, RAM, slice_size=1)
        cluster.set_memory_model(model)
        return cluster

    def test_sequential_consistency_publishes_data_first(self, assembler):
        data = run_message_passing(self._cluster(MemoryModel.SEQUENTIAL), assembler,
                                   PLAIN_WRITER, PLAIN_READER)
        assert data == 42

    def test_adversarial_model_exposes_the_missing_barrier(self, assembler):
        # The flag reaches memory before the data it was supposed to be guarding.
        data = run_message_passing(self._cluster(MemoryModel.ADVERSARIAL), assembler,
                                   PLAIN_WRITER, PLAIN_READER)
        assert data == 0

    def test_a_writer_side_barrier_alone_is_not_enough(self, assembler):
        # The writer now publishes data before the flag, but the reader's two loads are
        # still unordered with respect to each other, so it can read the data from
        # before the flag was ever set. Message passing needs a barrier on both sides.
        data = run_message_passing(self._cluster(MemoryModel.ADVERSARIAL), assembler,
                                   BARRIER_WRITER, PLAIN_READER)
        assert data == 0

    def test_barriers_on_both_sides_fix_it(self, assembler):
        data = run_message_passing(self._cluster(MemoryModel.ADVERSARIAL), assembler,
                                   BARRIER_WRITER, BARRIER_READER)
        assert data == 42

    def test_release_and_acquire_fix_it(self, assembler):
        # STLR publishes everything before it; LDAR stops later reads overtaking it.
        data = run_message_passing(self._cluster(MemoryModel.ADVERSARIAL), assembler,
                                   RELEASE_WRITER, ACQUIRE_READER)
        assert data == 42


LOCK = 0x8000
COUNTER = 0x8100
ITERATIONS = 15
CORES = 4

SPINLOCK = f'''
        movz x10, #{LOCK}
        movz x11, #{COUNTER}
        movz w12, #{ITERATIONS}
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
#: The same lock, released with a plain store. The counter update can still be sitting
#: in the buffer when the lock already looks free to another core.
BROKEN_RELEASE = SPINLOCK.replace('stlr  wzr, [x10]', 'str   wzr, [x10]')


class TestSpinlockUnderReordering:
    def _run(self, assembler, source, model, latency=None):
        cluster = Cluster(CORES, RAM, slice_size=1)
        cluster.set_memory_model(model, latency=latency)
        load(cluster, assembler, 0x1000, source)
        write(cluster, LOCK, 0)
        write(cluster, COUNTER, 0)
        for cpu_id in range(CORES):
            cluster.power_on(cpu_id, 0x1000)
        cluster.run(600000)
        return read(cluster, COUNTER)

    def test_a_correct_lock_survives_the_adversarial_model(self, assembler):
        assert self._run(assembler, SPINLOCK, MemoryModel.ADVERSARIAL) == CORES * ITERATIONS

    def test_a_plain_store_release_loses_updates(self, assembler):
        # Under sequential consistency this same code is indistinguishable from correct.
        assert self._run(assembler, BROKEN_RELEASE,
                         MemoryModel.SEQUENTIAL) == CORES * ITERATIONS
        assert self._run(assembler, BROKEN_RELEASE,
                         MemoryModel.ADVERSARIAL) < CORES * ITERATIONS


class TestDeviceMemoryIsNotBuffered:
    def test_device_writes_bypass_the_buffer(self, assembler):
        # A buffered write to a peripheral would arrive late, so Device memory must go
        # straight out. With the MMU off everything is Normal, so this checks the
        # predicate the translation path supplies rather than going through firmware.
        from armulator.armv6.memory_attributes import MemoryAttributes, MemType
        from armulator.armv6.address_descriptor import AddressDescriptor as Descriptor
        cluster = Cluster(1, RAM, slice_size=1)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        core = cluster.cores[0]

        descriptor = Descriptor()
        descriptor.memattrs = MemoryAttributes()
        descriptor.memattrs.type = MemType.DEVICE
        assert core._is_bufferable(descriptor) is False

        descriptor.memattrs.type = MemType.NORMAL
        assert core._is_bufferable(descriptor) is True


class TestBarriersDrain:
    @pytest.mark.parametrize('barrier', ['dmb sy', 'dsb sy', 'isb'])
    def test_each_barrier_empties_the_buffer(self, assembler, barrier):
        cluster = Cluster(1, RAM, slice_size=8)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        load(cluster, assembler, 0x1000,
             f'movz x10, #{X}\n movz w1, #1\n str w1, [x10]\n {barrier}\n b .')
        write(cluster, X, 0)
        cluster.power_on(0, 0x1000)
        # Step just past the barrier rather than to completion, so the end-of-run
        # drain cannot be what makes this pass.
        for _ in range(5):
            cluster.cores[0].emulate_cycle()
        assert len(cluster.cores[0].store_buffer) == 0
        assert read(cluster, X) == 1


ACQUIRE_ONLY_READER = 'spin: ldar w4, [x11]\n cbz w4, spin\n ldr w3, [x10]'


class TestLoadThenLoadReordering:
    """
    The reader side of message passing, with a writer that is correct throughout.

    Store buffering cannot produce this failure: the writer's barrier guarantees the
    data reaches memory before the flag, so no consistent snapshot shows the flag set
    without it. What breaks it is the reader's own two loads being unordered.
    """

    def _cluster(self, model):
        cluster = Cluster(2, RAM, slice_size=1)
        cluster.set_memory_model(model)
        return cluster

    def test_sequential_consistency_orders_the_loads(self, assembler):
        data = run_message_passing(self._cluster(MemoryModel.SEQUENTIAL), assembler,
                                   BARRIER_WRITER, PLAIN_READER)
        assert data == 42

    def test_store_buffering_alone_cannot_produce_it(self, assembler):
        # The relaxed model reorders stores but not loads, and the writer is correct,
        # so this failure is out of its reach. If this ever starts failing, the store
        # buffer has become able to reorder something it should not.
        data = run_message_passing(self._cluster(MemoryModel.RELAXED), assembler,
                                   BARRIER_WRITER, PLAIN_READER)
        assert data == 42

    def test_load_reordering_exposes_the_unbarriered_reader(self, assembler):
        data = run_message_passing(self._cluster(MemoryModel.ADVERSARIAL), assembler,
                                   BARRIER_WRITER, PLAIN_READER)
        assert data == 0

    def test_a_reader_side_barrier_fixes_it(self, assembler):
        data = run_message_passing(self._cluster(MemoryModel.ADVERSARIAL), assembler,
                                   BARRIER_WRITER, BARRIER_READER)
        assert data == 42

    def test_an_acquire_load_fixes_it(self, assembler):
        # LDAR orders everything after it, so the data load cannot be answered from
        # before the flag was read.
        data = run_message_passing(self._cluster(MemoryModel.ADVERSARIAL), assembler,
                                   BARRIER_WRITER, ACQUIRE_ONLY_READER)
        assert data == 42


class TestLoadReordererRules:
    def test_a_repeated_load_reads_fresh(self):
        from armulator.armv8.store_buffer import LoadReorderer
        reorderer = LoadReorderer(MemoryModel.ADVERSARIAL)
        reorderer.synchronize(10)
        # First read of an address looks back; a re-read is a loop and must not, or a
        # spin would never observe the value it is waiting for.
        assert reorderer.read_time(0x100, 50) == 10
        assert reorderer.read_time(0x100, 51) == 51

    def test_a_new_address_looks_back_to_the_last_barrier(self):
        from armulator.armv8.store_buffer import LoadReorderer
        reorderer = LoadReorderer(MemoryModel.ADVERSARIAL)
        reorderer.synchronize(10)
        reorderer.read_time(0x100, 50)
        assert reorderer.read_time(0x200, 51) == 10

    def test_a_barrier_moves_the_window_forward(self):
        from armulator.armv8.store_buffer import LoadReorderer
        reorderer = LoadReorderer(MemoryModel.ADVERSARIAL)
        reorderer.synchronize(10)
        reorderer.read_time(0x100, 50)
        reorderer.synchronize(60)
        assert reorderer.read_time(0x100, 61) == 60

    def test_the_relaxed_model_does_not_reorder_loads(self):
        from armulator.armv8.store_buffer import LoadReorderer
        reorderer = LoadReorderer(MemoryModel.RELAXED)
        assert reorderer.reordering is False
        assert reorderer.read_time(0x100, 50) == 50


class TestACoreSeesItsOwnWrites:
    def test_a_write_then_a_first_read_of_that_address(self, assembler):
        # The reordering rule would answer this load from before the write, so the
        # core's own store has to override it. Getting this wrong breaks every
        # single-threaded program.
        cluster = Cluster(1, RAM, slice_size=1)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        load(cluster, assembler, 0x1000,
             f'movz x10, #{X}\n movz w1, #0x42\n str w1, [x10]\n dmb sy\n'
             ' ldr w2, [x10]\n b .')
        write(cluster, X, 0)
        cluster.power_on(0, 0x1000)
        cluster.run(500)
        assert cluster.cores[0].registers.get_x(2) == 0x42

    def test_a_long_sequence_of_writes_and_reads(self, assembler):
        cluster = Cluster(1, RAM, slice_size=1)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        load(cluster, assembler, 0x1000, f'''
                movz x10, #{X}
                movz w12, #5
                movz w13, #0
        loop:   add  w13, w13, #1
                str  w13, [x10]
                ldr  w14, [x10]
                sub  w12, w12, #1
                cbnz w12, loop
                b .
        ''')
        write(cluster, X, 0)
        cluster.power_on(0, 0x1000)
        cluster.run(5000)
        # Every read must have returned what was just written.
        assert cluster.cores[0].registers.get_x(14) == 5


def run_load_buffering(cluster, assembler, barrier=''):
    """
    Each core reads one location then writes the other. Sequential consistency forbids
    both reads returning 1 - that would need each write to have happened before the read
    that precedes it in program order.
    """
    load(cluster, assembler, 0x1000,
         f'movz x10, #{X}\n movz x11, #{Y}\n ldr w2, [x10]\n {barrier}\n'
         f' movz w1, #1\n str w1, [x11]\n b .')
    load(cluster, assembler, 0x2000,
         f'movz x10, #{X}\n movz x11, #{Y}\n ldr w2, [x11]\n {barrier}\n'
         f' movz w1, #1\n str w1, [x10]\n b .')
    write(cluster, X, 0)
    write(cluster, Y, 0)
    cluster.power_on(0, 0x1000)
    cluster.power_on(1, 0x2000)
    cluster.run(20000)
    return cluster.cores[0].registers.get_x(2), cluster.cores[1].registers.get_x(2)


class TestLoadThenStoreReordering:
    """
    Load buffering: a store becoming visible before a load that precedes it.

    The store buffer cannot produce this - buffering makes stores land later, and this
    needs one to be seen earlier. It is modelled from the other side: the load is
    performed late, which is the same reordering and is a direction a forward simulation
    can actually take.
    """

    def _cluster(self, model):
        cluster = Cluster(2, RAM, slice_size=1)
        cluster.set_memory_model(model)
        return cluster

    def test_sequential_consistency_forbids_both_ones(self, assembler):
        assert run_load_buffering(self._cluster(MemoryModel.SEQUENTIAL), assembler) != (1, 1)

    def test_store_buffering_alone_cannot_produce_it(self, assembler):
        # Delaying stores can never make one visible earlier, so this outcome is out of
        # the relaxed model's reach by construction.
        assert run_load_buffering(self._cluster(MemoryModel.RELAXED), assembler) != (1, 1)

    def test_the_adversarial_model_produces_it(self, assembler):
        assert run_load_buffering(self._cluster(MemoryModel.ADVERSARIAL), assembler) == (1, 1)

    def test_a_barrier_between_the_load_and_the_store_fixes_it(self, assembler):
        result = run_load_buffering(self._cluster(MemoryModel.ADVERSARIAL), assembler,
                                    barrier='dmb sy')
        assert result != (1, 1)


class TestReorderDirectionIsDecidedByContext:
    """
    A load can be reordered in either direction, but not both at once - the two move it
    opposite ways. Which applies is decided by what follows the load.
    """

    def test_a_load_followed_by_a_store_moves_late(self, assembler):
        # Load buffering: the store gets ahead of the load.
        assert run_load_buffering(self._cluster(), assembler) == (1, 1)

    def test_a_load_followed_only_by_loads_moves_early(self, assembler):
        # Message passing: the data load is answered from before the flag was set.
        data = run_message_passing(self._cluster(), assembler,
                                   BARRIER_WRITER, PLAIN_READER)
        assert data == 0

    def _cluster(self):
        cluster = Cluster(2, RAM, slice_size=1)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        return cluster


class TestCoherence:
    def test_a_core_never_sees_its_own_writes_go_backwards(self, assembler):
        # Two stores to the same address may not be reordered against each other -
        # coherence forbids it, and breaking it would make a core observe its own
        # writes out of order rather than modelling weak memory.
        cluster = Cluster(1, RAM, slice_size=1)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        load(cluster, assembler, 0x1000, f'''
                movz x10, #{X}
                movz w12, #5
                movz w13, #0
        loop:   add  w13, w13, #1
                str  w13, [x10]
                ldr  w14, [x10]
                sub  w12, w12, #1
                cbnz w12, loop
                b .
        ''')
        write(cluster, X, 0)
        cluster.power_on(0, 0x1000)
        cluster.run(5000)
        assert cluster.cores[0].registers.get_x(14) == 5
        assert read(cluster, X) == 5

    def test_stores_to_one_address_reach_memory_in_program_order(self):
        buffer = StoreBuffer(model=MemoryModel.ADVERSARIAL)
        buffer.push(0x1000, 4, 1)
        buffer.push(0x2000, 4, 2)
        buffer.push(0x1000, 4, 3)
        written = []
        buffer.drain(lambda a, s, v: written.append((a, v)))
        # The unrelated address may overtake, but 0x1000's two writes keep their order.
        order = [value for address, value in written if address == 0x1000]
        assert order == [1, 3]


class TestDeferredLoadsAreSafe:
    def test_using_a_value_settles_it_first(self, assembler):
        # A deferred load must never let the program observe a provisional value, so
        # reading the register resolves it.
        cluster = Cluster(1, RAM, slice_size=1)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        load(cluster, assembler, 0x1000,
             f'movz x10, #{X}\n movz w1, #7\n str w1, [x10]\n dmb sy\n'
             ' ldr w2, [x10]\n add w3, w2, #1\n b .')
        write(cluster, X, 0)
        cluster.power_on(0, 0x1000)
        cluster.run(2000)
        assert cluster.cores[0].registers.get_x(2) == 7
        assert cluster.cores[0].registers.get_x(3) == 8

    def test_a_store_to_the_same_address_settles_it(self, assembler):
        # A load cannot be reordered past a store to the location it read.
        cluster = Cluster(1, RAM, slice_size=1)
        cluster.set_memory_model(MemoryModel.ADVERSARIAL)
        load(cluster, assembler, 0x1000,
             f'movz x10, #{X}\n movz w1, #7\n str w1, [x10]\n dmb sy\n'
             ' ldr w2, [x10]\n movz w4, #99\n str w4, [x10]\n b .')
        write(cluster, X, 0)
        cluster.power_on(0, 0x1000)
        cluster.run(2000)
        assert cluster.cores[0].registers.get_x(2) == 7
