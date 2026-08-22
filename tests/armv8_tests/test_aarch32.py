"""
AArch32 execution at EL0.

The instruction semantics are the ARMv6 core's and are tested there; what is tested here
is the adapter — that AArch32 state maps onto the AArch64 register file correctly, and
that the boundary between the two execution states works in both directions.
"""

import pytest
from keystone import (
    KS_ARCH_ARM,
    KS_ARCH_ARM64,
    KS_MODE_ARM,
    KS_MODE_LITTLE_ENDIAN,
    KS_MODE_THUMB,
    Ks,
)

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.armv6.enums import InstrSet as A32InstrSet
from armulator.armv8.arm_v8 import ArmV8
from armulator.armv8.enums import EL

RAM = [{'mem_type': 'RAM', 'beginning': 0x0, 'end': 0x40000}]
VBAR_EL1 = 0x20000
#: The vector group an exception from a lower EL running AArch32 enters through.
LOWER_EL_A32 = 0x600


@pytest.fixture(scope='module')
def a32():
    return Ks(KS_ARCH_ARM, KS_MODE_ARM)


@pytest.fixture(scope='module')
def t32():
    return Ks(KS_ARCH_ARM, KS_MODE_THUMB)


@pytest.fixture(scope='module')
def a64():
    return Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)


@pytest.fixture
def processor():
    cpu = ArmV8(RAM)
    cpu.take_reset()
    cpu.psci_handler = None
    cpu.registers.vbar[EL.EL1] = VBAR_EL1
    return cpu


def load(processor, address, assembler, source):
    code, _ = assembler.asm(source, address)
    for offset, byte in enumerate(bytes(code)):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address + offset
        processor.mem[descriptor, 1] = byte


def read(processor, address, size=4):
    descriptor = AddressDescriptor()
    descriptor.paddress.physicaladdress = address
    return processor.mem[descriptor, size]


def run(processor, count):
    for _ in range(count):
        processor.emulate_cycle()


class TestEnteringAArch32:
    def test_execution_state_is_recorded_in_pstate(self, processor, a32):
        load(processor, 0x1000, a32, 'mov r0, #1')
        processor.enter_aarch32(0x1000)
        assert processor.registers.using_aarch32() is True
        assert processor.registers.pstate.el == EL.EL0

    def test_instructions_execute(self, processor, a32):
        load(processor, 0x1000, a32, 'mov r0, #5\n add r1, r0, #3')
        processor.enter_aarch32(0x1000)
        run(processor, 2)
        assert processor.registers.get_x(0, 32) == 5
        assert processor.registers.get_x(1, 32) == 8

    def test_an_interworking_address_has_its_low_bit_stripped(self, processor, t32):
        load(processor, 0x1100, t32, 'movs r0, #1')
        processor.enter_aarch32(0x1101, thumb=True)
        assert processor.aarch32.pc == 0x1100
        assert processor.aarch32.instruction_set is A32InstrSet.THUMB


class TestRegisterMapping:
    def test_aarch32_registers_are_the_low_halves_of_the_aarch64_ones(self, processor, a32):
        # This is what lets a 64-bit kernel read a 32-bit application's arguments
        # straight out of X0-X7.
        load(processor, 0x1000, a32, 'mov r3, #0x42')
        processor.enter_aarch32(0x1000)
        run(processor, 1)
        assert processor.registers.get_x(3) == 0x42

    def test_a_64_bit_value_is_truncated_when_read_as_32(self, processor, a32):
        processor.registers.set_x(5, 0xAAAABBBB11112222)
        load(processor, 0x1000, a32, 'mov r0, r5')
        processor.enter_aarch32(0x1000)
        run(processor, 1)
        assert processor.registers.get_x(0, 32) == 0x11112222

    def test_condition_flags_are_shared_with_pstate(self, processor, a32):
        load(processor, 0x1000, a32, 'mov r0, #5\n cmp r0, #5')
        processor.enter_aarch32(0x1000)
        run(processor, 2)
        # The AArch32 CPSR and PSTATE are the same bits under two names.
        assert processor.registers.pstate.z == 1
        assert processor.aarch32.registers.cpsr.z == 1

    def test_the_pc_reads_with_its_pipeline_offset(self, processor, a32):
        # Reading the PC in A32 gives the instruction address plus eight, which AArch64
        # does not do and which hand-written ARM assembly depends on.
        load(processor, 0x1000, a32, 'mov r0, pc')
        processor.enter_aarch32(0x1000)
        run(processor, 1)
        assert processor.registers.get_x(0, 32) == 0x1008


class TestConditionalExecution:
    def test_conditional_instructions_are_honoured(self, processor, a32):
        # Per-instruction condition codes exist only in AArch32.
        load(processor, 0x1000, a32,
             'mov r0, #5\n cmp r0, #5\n moveq r1, #100\n movne r1, #200')
        processor.enter_aarch32(0x1000)
        run(processor, 4)
        assert processor.registers.get_x(1, 32) == 100

    def test_a_failing_condition_skips_the_instruction(self, processor, a32):
        load(processor, 0x1000, a32, 'mov r0, #5\n cmp r0, #4\n moveq r1, #100')
        processor.enter_aarch32(0x1000)
        run(processor, 3)
        assert processor.registers.get_x(1, 32) == 0


class TestInterworking:
    def test_bx_switches_to_thumb(self, processor, a32, t32):
        load(processor, 0x1000, a32, 'ldr r1, =0x1101\n bx r1')
        load(processor, 0x1100, t32, 'movs r2, #9\n adds r2, #1')
        processor.enter_aarch32(0x1000)
        run(processor, 2)
        assert processor.aarch32.instruction_set is A32InstrSet.THUMB
        assert processor.aarch32.pc == 0x1100
        run(processor, 2)
        assert processor.registers.get_x(2, 32) == 10


class TestMemoryIsShared:
    def test_a_store_is_visible_to_the_aarch64_side(self, processor, a32):
        load(processor, 0x1000, a32, 'mov r0, #0x37\n mov r1, #0x200\n str r0, [r1]')
        processor.enter_aarch32(0x1000)
        run(processor, 3)
        assert read(processor, 0x200) == 0x37

    def test_a_loop_runs_to_completion(self, processor, a32):
        load(processor, 0x1000, a32, '''
                mov r0, #0
                mov r1, #5
        loop:   add r0, r0, r1
                subs r1, r1, #1
                bne loop
        ''')
        processor.enter_aarch32(0x1000)
        run(processor, 40)
        assert processor.registers.get_x(0, 32) == 15


class TestExceptionsFromAArch32:
    def test_svc_enters_aarch64_at_el1(self, processor, a32, a64):
        load(processor, 0x1000, a32, 'mov r0, #42\n svc #0x99')
        load(processor, VBAR_EL1 + LOWER_EL_A32, a64, 'b .')
        processor.enter_aarch32(0x1000)
        run(processor, 2)
        assert processor.registers.pstate.el == EL.EL1
        # Handling an AArch32 exception happens in AArch64 - there is no AArch32
        # exception level here.
        assert processor.registers.using_aarch32() is False

    def test_it_uses_the_lower_el_aarch32_vector_group(self, processor, a32, a64):
        load(processor, 0x1000, a32, 'svc #1')
        load(processor, VBAR_EL1 + LOWER_EL_A32, a64, 'b .')
        processor.enter_aarch32(0x1000)
        run(processor, 1)
        assert processor.registers.get_pc() == VBAR_EL1 + LOWER_EL_A32

    def test_the_syndrome_says_the_call_came_from_aarch32(self, processor, a32, a64):
        # EC 0x11 rather than 0x15, so a handler can tell without reading the SPSR.
        load(processor, 0x1000, a32, 'svc #0x99')
        load(processor, VBAR_EL1 + LOWER_EL_A32, a64, 'b .')
        processor.enter_aarch32(0x1000)
        run(processor, 1)
        esr = processor.registers.esr[EL.EL1]
        assert esr >> 26 == 0x11
        assert esr & 0xFFFF == 0x99

    def test_the_return_address_is_the_following_instruction(self, processor, a32, a64):
        load(processor, 0x1000, a32, 'mov r0, #1\n svc #1\n mov r0, #7')
        load(processor, VBAR_EL1 + LOWER_EL_A32, a64, 'b .')
        processor.enter_aarch32(0x1000)
        run(processor, 2)
        assert processor.registers.elr[EL.EL1] == 0x1008

    def test_the_kernel_can_read_the_arguments(self, processor, a32, a64):
        load(processor, 0x1000, a32, 'mov r0, #42\n svc #1')
        load(processor, VBAR_EL1 + LOWER_EL_A32, a64, 'mov x12, x0\n eret')
        processor.enter_aarch32(0x1000)
        run(processor, 3)
        assert processor.registers.get_x(12) == 42

    def test_an_undefined_instruction_faults_to_el1(self, processor, a64):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = 0x1000
        processor.mem[descriptor, 4] = 0xE7FFFFFF     # permanently undefined in A32
        load(processor, VBAR_EL1 + LOWER_EL_A32, a64, 'b .')
        processor.enter_aarch32(0x1000)
        run(processor, 1)
        assert processor.registers.pstate.el == EL.EL1
        assert processor.registers.elr[EL.EL1] == 0x1000


class TestReturningToAArch32:
    def test_eret_resumes_where_it_left_off(self, processor, a32, a64):
        load(processor, 0x1000, a32, 'mov r0, #42\n svc #1\n mov r0, #7')
        load(processor, VBAR_EL1 + LOWER_EL_A32, a64, 'eret')
        processor.enter_aarch32(0x1000)
        run(processor, 3)          # mov, svc, eret
        assert processor.registers.using_aarch32() is True
        assert processor.aarch32.pc == 0x1008
        run(processor, 1)
        assert processor.registers.get_x(0, 32) == 7

    def test_eret_restores_the_thumb_state(self, processor, t32, a64):
        # The T bit rides in the SPSR; without it the return would resume in A32 and
        # misdecode everything after it.
        load(processor, 0x1100, t32, 'movs r0, #1\n svc #1\n movs r0, #9')
        load(processor, VBAR_EL1 + LOWER_EL_A32, a64, 'eret')
        processor.enter_aarch32(0x1101, thumb=True)
        run(processor, 3)
        assert processor.aarch32.instruction_set is A32InstrSet.THUMB
        run(processor, 1)
        assert processor.registers.get_x(0, 32) == 9

    def test_a_thumb_call_returns_to_the_right_address(self, processor, t32, a64):
        # A T32 SVC is two bytes, not four.
        load(processor, 0x1100, t32, 'movs r0, #1\n svc #1\n movs r0, #9')
        load(processor, VBAR_EL1 + LOWER_EL_A32, a64, 'b .')
        processor.enter_aarch32(0x1100, thumb=True)
        run(processor, 2)
        assert processor.registers.elr[EL.EL1] == 0x1104
