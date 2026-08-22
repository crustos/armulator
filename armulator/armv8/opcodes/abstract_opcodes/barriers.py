from armulator.armv8.enums import MBReqDomain, MBReqTypes
from armulator.armv8.opcodes.opcode import Opcode

_DOMAINS = {
    0b00: MBReqDomain.OUTER_SHAREABLE,
    0b01: MBReqDomain.NONSHAREABLE,
    0b10: MBReqDomain.INNER_SHAREABLE,
    0b11: MBReqDomain.FULL_SYSTEM,
}
_TYPES = {0b01: MBReqTypes.READS, 0b10: MBReqTypes.WRITES, 0b11: MBReqTypes.ALL}


class Barrier(Opcode):
    """
    DMB, DSB and ISB. The emulator executes strictly in order with no store buffer, so
    barriers have nothing to enforce - but they must decode and retire, since real
    firmware puts them between every peripheral access.
    """

    def __init__(self, instruction, kind, crm):
        super().__init__(instruction)
        self.kind = kind
        self.crm = crm

    @property
    def domain(self):
        return _DOMAINS[(self.crm >> 2) & 0b11]

    @property
    def types(self):
        return _TYPES.get(self.crm & 0b11, MBReqTypes.ALL)

    def execute(self, processor):
        if self.kind == 'isb':
            processor.instruction_synchronization_barrier()
        elif self.kind == 'dsb':
            processor.data_synchronization_barrier(self.domain, self.types)
        else:
            processor.data_memory_barrier(self.domain, self.types)
