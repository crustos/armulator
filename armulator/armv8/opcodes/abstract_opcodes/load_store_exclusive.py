from armulator.armv8.enums import MemOp
from armulator.armv8.opcodes.opcode import Opcode


class LoadStoreExclusive(Opcode):
    """
    LDXR/STXR and the acquire/release forms LDAR/STLR.

    LDXR takes a reservation and STXR only succeeds while that reservation survives.
    The status register reports 0 for success and 1 for failure, which is what lets a
    spinlock retry rather than silently believing it holds a lock it does not.

    LDAR and STLR carry ordering semantics but no reservation, and under a relaxed
    memory model those semantics matter. A release store must not become visible before
    anything that preceded it, so it drains the store buffer first; an acquire load must
    not be overtaken by anything that follows it. That pairing is what makes the flag
    published by one core imply the data behind it on another.
    """

    def __init__(self, instruction, t, t2, n, s, memop, datasize, pair, exclusive):
        super().__init__(instruction)
        self.t = t
        self.t2 = t2
        self.n = n
        self.s = s
        self.memop = memop
        self.datasize = datasize
        self.pair = pair
        self.exclusive = exclusive

    @property
    def size(self):
        return self.datasize // 8

    def execute(self, processor):
        if self.n == 31:
            processor.check_sp_alignment()
            address = processor.registers.get_sp()
        else:
            address = processor.registers.get_x(self.n)

        size = self.size
        monitor = processor.exclusive_monitor

        if self.memop == MemOp.STORE:
            # Release ordering: everything already issued becomes visible before this
            # store does. Without this a reader could see the flag and miss the data.
            processor.drain_store_buffer()
            processor.synchronize_reads()
            if self.exclusive:
                if not monitor.check_and_clear(processor.cpu_id, address):
                    # The reservation was lost, so the store must not happen at all.
                    processor.registers.set_x(self.s, 1, 32)
                    return
            processor.mem_set(address, size,
                              processor.registers.get_x(self.t, self.datasize))
            if self.pair:
                processor.mem_set(address + size, size,
                                  processor.registers.get_x(self.t2, self.datasize))
            if self.exclusive:
                processor.registers.set_x(self.s, 0, 32)
            # An exclusive or release store must be visible when it retires, not queued
            # behind the buffer, so it is pushed straight out.
            processor.drain_store_buffer()
            return

        # An exclusive load must observe memory rather than this core's pending writes
        # alone, so the buffer is settled before the reservation is taken. An acquire
        # load is also an ordering point: nothing after it may be answered from before
        # it, which is what stops a speculated load overtaking the flag it guards.
        processor.drain_store_buffer()
        processor.synchronize_reads()
        if self.exclusive:
            monitor.reserve(processor.cpu_id, address)
        processor.registers.set_x(self.t, processor.mem_get(address, size), self.datasize)
        if self.pair:
            processor.registers.set_x(
                self.t2, processor.mem_get(address + size, size), self.datasize
            )
