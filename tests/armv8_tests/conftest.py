import pytest
from keystone import KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN, Ks

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.armv8.arm_v8 import ArmV8

RAM = [{'mem_type': 'RAM', 'beginning': 0x0, 'end': 0x10000}]
LOAD_ADDRESS = 0x1000


@pytest.fixture(scope='session')
def assembler():
    return Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)


class A64Runner:
    """
    Assembles a snippet, drops it into RAM and steps the processor over it.
    """

    def __init__(self, assembler):
        self.assembler = assembler

    def build(self, source, load_address=LOAD_ADDRESS):
        proc = ArmV8(RAM)
        proc.take_reset()
        code, _ = self.assembler.asm(source, load_address)
        code = bytes(code)
        for offset, byte in enumerate(code):
            descriptor = AddressDescriptor()
            descriptor.paddress.physicaladdress = load_address + offset
            proc.mem[descriptor, 1] = byte
        proc.registers.branch_to(load_address)
        return proc, len(code) // 4

    @staticmethod
    def enable_fp(proc):
        """
        Set CPACR_EL1.FPEN so SIMD and floating point instructions may execute.

        CPACR_EL1 resets to trapping everything, so this mirrors what bare-metal
        startup code has to do before touching a vector register.
        """
        proc.registers.set_system_register(0b11, 0b000, 0b0001, 0b0000, 0b010, 0b11 << 20)

    def run(self, source, steps=None, load_address=LOAD_ADDRESS, setup=None, fp=False):
        """
        Assemble, load and step.

        By default one instruction is executed per instruction assembled, which suits
        straight-line code. Snippets that branch backwards need an explicit ``steps``
        budget, since they execute more instructions than they contain.
        """
        proc, instructions = self.build(source, load_address)
        if fp:
            self.enable_fp(proc)
        if setup is not None:
            setup(proc)
        for _ in range(steps if steps is not None else instructions):
            proc.emulate_cycle()
        return proc

    #: Calling the runner directly is the common case.
    __call__ = run


@pytest.fixture
def run_a64(assembler):
    return A64Runner(assembler)
