"""
EL2 and EL3: exception routing, security state and stage 2 translation.
"""

import pytest
from keystone import KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN, Ks

from armulator.armv6.address_descriptor import AddressDescriptor
from armulator.armv8.arm_exceptions import DataAbortException
from armulator.armv8.arm_v8 import ArmV8
from armulator.armv8.enums import EL

RAM = [{'mem_type': 'RAM', 'beginning': 0x0, 'end': 0x1000000}]
VBAR = {EL.EL1: 0x10000, EL.EL2: 0x20000, EL.EL3: 0x30000}

VALID = 1
TABLE = 2
ACCESS_FLAG = 1 << 10
INNER_SHAREABLE = 3 << 8
AP_EL0_RW = 0b01 << 6
S2_READ = 1 << 6
S2_WRITE = 1 << 7

# System register encodings used throughout.
HCR_EL2 = (0b11, 0b100, 0b0001, 0b0001, 0b000)
SCR_EL3 = (0b11, 0b110, 0b0001, 0b0001, 0b000)
VTCR_EL2 = (0b11, 0b100, 0b0010, 0b0001, 0b010)
VTTBR_EL2 = (0b11, 0b100, 0b0010, 0b0001, 0b000)
HPFAR_EL2 = (0b11, 0b100, 0b0110, 0b0000, 0b100)
MAIR_EL1 = (0b11, 0b000, 0b1010, 0b0010, 0b000)
TCR_EL1 = (0b11, 0b000, 0b0010, 0b0000, 0b010)
TTBR0_EL1 = (0b11, 0b000, 0b0010, 0b0000, 0b000)

#: HCR_EL2.RW - EL1 is AArch64. Always set, or the value would describe a 32-bit guest.
HCR_RW = 1 << 31


@pytest.fixture(scope='module')
def assembler():
    return Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)


def make_processor(highest_el=EL.EL3):
    processor = ArmV8(RAM, highest_el=highest_el)
    processor.take_reset()
    processor.psci_handler = None
    for level, base in VBAR.items():
        processor.registers.vbar[level] = base
    return processor


def load(processor, assembler, address, source):
    code, _ = assembler.asm(source, address)
    for offset, byte in enumerate(bytes(code)):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address + offset
        processor.mem[descriptor, 1] = byte


def step(processor, address, count=1):
    processor.registers.branch_to(address)
    processor.registers.branch_taken = False
    for _ in range(count):
        processor.emulate_cycle()


class TestCallRouting:
    @pytest.mark.parametrize('instruction, source, expected', [
        ('svc #1', EL.EL0, EL.EL1),
        ('svc #1', EL.EL1, EL.EL1),
        ('hvc #1', EL.EL1, EL.EL2),
        ('hvc #1', EL.EL2, EL.EL2),
        ('smc #1', EL.EL1, EL.EL3),
        ('smc #1', EL.EL2, EL.EL3),
    ])
    def test_each_call_reaches_its_own_level(self, assembler, instruction, source, expected):
        processor = make_processor()
        processor.registers.pstate.el = source
        load(processor, assembler, 0x1000, instruction)
        step(processor, 0x1000)
        assert processor.registers.pstate.el == expected

    def test_an_exception_never_targets_a_lower_level(self, assembler):
        # An SVC at EL2 cannot drop to EL1 to be handled.
        processor = make_processor()
        processor.registers.pstate.el = EL.EL2
        load(processor, assembler, 0x1000, 'svc #1')
        step(processor, 0x1000)
        assert processor.registers.pstate.el == EL.EL2

    def test_smc_falls_back_when_there_is_no_el3(self, assembler):
        processor = make_processor(highest_el=EL.EL2)
        processor.registers.pstate.el = EL.EL1
        load(processor, assembler, 0x1000, 'smc #1')
        step(processor, 0x1000)
        assert processor.registers.pstate.el == EL.EL2

    def test_the_vector_group_reflects_where_it_came_from(self, assembler):
        # An exception from a lower level uses the +0x400 group; from the same level
        # with SP_ELx selected, the +0x200 group.
        processor = make_processor()
        processor.registers.pstate.el = EL.EL1
        load(processor, assembler, 0x1000, 'hvc #1')
        step(processor, 0x1000)
        assert processor.registers.get_pc() == VBAR[EL.EL2] + 0x400

        processor = make_processor()
        processor.registers.pstate.el = EL.EL2
        load(processor, assembler, 0x1000, 'hvc #1')
        step(processor, 0x1000)
        assert processor.registers.get_pc() == VBAR[EL.EL2] + 0x200


class TestInterruptRouting:
    def _deliver(self, hcr=0, scr=0, source=EL.EL1):
        processor = make_processor()
        processor.registers.pstate.el = source
        processor.registers.pstate.i = 0
        processor.registers.set_system_register(*HCR_EL2, HCR_RW | hcr)
        processor.registers.set_system_register(*SCR_EL3, (1 << 10) | scr)
        processor.registers.branch_to(0x1000)
        processor.registers.branch_taken = False
        processor.take_physical_irq_exception()
        return processor.registers.pstate.el

    def test_by_default_an_irq_goes_to_el1(self):
        assert self._deliver() == EL.EL1

    def test_hcr_imo_gives_the_irq_to_the_hypervisor(self):
        assert self._deliver(hcr=1 << 4) == EL.EL2

    def test_scr_irq_gives_it_to_secure_firmware(self):
        assert self._deliver(scr=1 << 1) == EL.EL3

    def test_el3_wins_over_el2(self):
        assert self._deliver(hcr=1 << 4, scr=1 << 1) == EL.EL3

    def test_fiq_routes_on_its_own_bit(self):
        processor = make_processor()
        processor.registers.pstate.el = EL.EL1
        processor.registers.pstate.f = 0
        processor.registers.set_system_register(*HCR_EL2, HCR_RW | (1 << 3))
        processor.registers.branch_to(0x1000)
        processor.registers.branch_taken = False
        processor.take_physical_fiq_exception()
        assert processor.registers.pstate.el == EL.EL2


class TestTrapGeneralExceptions:
    def test_tge_routes_el0_exceptions_to_el2(self, assembler):
        # With TGE set the hypervisor runs an application directly, with no guest
        # kernel underneath it to field the call.
        processor = make_processor()
        processor.registers.pstate.el = EL.EL0
        processor.registers.set_system_register(*HCR_EL2, HCR_RW | (1 << 27))
        load(processor, assembler, 0x1000, 'svc #5')
        step(processor, 0x1000)
        assert processor.registers.pstate.el == EL.EL2

    def test_without_tge_it_goes_to_el1(self, assembler):
        processor = make_processor()
        processor.registers.pstate.el = EL.EL0
        load(processor, assembler, 0x1000, 'svc #5')
        step(processor, 0x1000)
        assert processor.registers.pstate.el == EL.EL1


class TestExceptionReturn:
    def test_eret_drops_from_el3_to_el1(self, assembler):
        # The boot path: firmware sets up the level it wants and returns into it.
        processor = make_processor()
        processor.registers.pstate.el = EL.EL3
        # SPSR describing EL1 with SP_EL1 selected and everything masked.
        processor.registers.spsr[EL.EL3] = (0b1111 << 6) | (0b01 << 2) | 1
        processor.registers.elr[EL.EL3] = 0x5000
        load(processor, assembler, 0x1000, 'eret')
        load(processor, assembler, 0x5000, 'movz x9, #0x77')
        step(processor, 0x1000)
        assert processor.registers.pstate.el == EL.EL1
        assert processor.registers.get_pc() == 0x5000
        processor.emulate_cycle()
        assert processor.registers.get_x(9) == 0x77

    def test_a_full_round_trip_returns_to_the_caller(self, assembler):
        processor = make_processor()
        processor.registers.pstate.el = EL.EL1
        load(processor, assembler, 0x1000, 'hvc #1\n movz x9, #0x42')
        load(processor, assembler, VBAR[EL.EL2] + 0x400, 'eret')
        step(processor, 0x1000)
        assert processor.registers.pstate.el == EL.EL2
        processor.emulate_cycle()          # the ERET
        assert processor.registers.pstate.el == EL.EL1
        processor.emulate_cycle()
        assert processor.registers.get_x(9) == 0x42


class TestSecurityState:
    def test_reset_is_to_secure_state(self):
        processor = make_processor()
        assert processor.registers.secure is True

    def test_clearing_scr_ns_leaves_secure_state(self):
        processor = make_processor()
        processor.registers.pstate.el = EL.EL1
        processor.registers.set_system_register(*SCR_EL3, (1 << 10) | 1)
        assert processor.registers.secure is False

    def test_el3_is_always_secure(self):
        processor = make_processor()
        processor.registers.set_system_register(*SCR_EL3, (1 << 10) | 1)
        processor.registers.pstate.el = EL.EL3
        assert processor.registers.secure is True


class TestTranslationRegimes:
    def test_each_level_uses_its_own_control_registers(self):
        processor = make_processor()
        for level in (EL.EL1, EL.EL2, EL.EL3):
            processor.registers.pstate.el = level
            assert processor.registers.regime() == level

    def test_el0_translates_through_el1(self):
        processor = make_processor()
        processor.registers.pstate.el = EL.EL0
        assert processor.registers.regime() == EL.EL1

    def test_enabling_the_mmu_at_one_level_does_not_affect_another(self):
        processor = make_processor()
        processor.registers.pstate.el = EL.EL1
        processor.registers.sctlr_el1 |= 1
        assert processor.registers.mmu_enabled is True
        processor.registers.pstate.el = EL.EL2
        assert processor.registers.mmu_enabled is False


class Stage2Fixture:
    """
    A guest mapping VA 0x40000000 to IPA 0x200000, with stage 2 sending that IPA
    somewhere else entirely.
    """

    GUEST_VA = 0x40000000
    GUEST_IPA = 0x200000
    REAL_PA = 0x900000

    def __init__(self):
        self.processor = make_processor()
        self.next_free = 0x100000
        self.stage1_root = self._build_stage1()
        self.stage2_leaf = None
        self.stage2_root = self._build_stage2()
        self._configure()

    def write64(self, address, value):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address
        self.processor.mem[descriptor, 8] = value

    def read64(self, address):
        descriptor = AddressDescriptor()
        descriptor.paddress.physicaladdress = address
        return self.processor.mem[descriptor, 8]

    def allocate(self):
        table = self.next_free
        self.next_free += 0x1000
        for offset in range(0, 0x1000, 8):
            self.write64(table + offset, 0)
        return table

    def _build_stage1(self):
        root, l1, l2, l3 = (self.allocate() for _ in range(4))
        self.write64(root, l1 | TABLE | VALID)
        self.write64(l1 + 1 * 8, l2 | TABLE | VALID)
        self.write64(l2, l3 | TABLE | VALID)
        self.write64(l3, self.GUEST_IPA | TABLE | VALID | ACCESS_FLAG
                     | INNER_SHAREABLE | AP_EL0_RW)
        return root

    def _build_stage2(self, permissions=S2_READ | S2_WRITE):
        root, l1, l2 = (self.allocate() for _ in range(3))
        ipa = self.GUEST_IPA
        self.write64(root + ((ipa >> 30) & 0x1FF) * 8, l1 | TABLE | VALID)
        self.write64(l1 + ((ipa >> 21) & 0x1FF) * 8, l2 | TABLE | VALID)
        self.stage2_leaf = l2 + ((ipa >> 12) & 0x1FF) * 8
        self.set_stage2_permissions(permissions)
        return root

    def set_stage2_permissions(self, permissions):
        self.write64(self.stage2_leaf, self.REAL_PA | TABLE | VALID | ACCESS_FLAG
                     | INNER_SHAREABLE | permissions)
        self.processor.mmu.flush()

    def unmap_stage2(self):
        self.write64(self.stage2_leaf, 0)
        self.processor.mmu.flush()

    def _configure(self):
        registers = self.processor.registers
        registers.set_system_register(*MAIR_EL1, 0xFF)
        registers.set_system_register(*TCR_EL1, 16 | (16 << 16) | (0b10 << 30))
        registers.set_system_register(*TTBR0_EL1, self.stage1_root)
        registers.sctlr_el1 |= 1
        # T0SZ 16 for a 48-bit intermediate address, SL0 01 to start at level 1.
        registers.set_system_register(*VTCR_EL2, 16 | (0b01 << 6))
        registers.set_system_register(*VTTBR_EL2, self.stage2_root)
        self.write64(self.REAL_PA, 0xC0FFEE)
        self.write64(self.GUEST_IPA, 0xBADBAD)

    def enable_stage2(self, enabled=True):
        self.processor.registers.set_system_register(
            *HCR_EL2, HCR_RW | (1 if enabled else 0)
        )
        self.processor.mmu.flush()


@pytest.fixture
def stage2():
    return Stage2Fixture()


class TestStageTwoTranslation:
    def test_without_stage2_the_guest_sees_its_own_addresses(self, stage2):
        stage2.enable_stage2(False)
        descriptor = stage2.processor.translate_address(stage2.GUEST_VA)
        assert descriptor.paddress.physicaladdress == stage2.GUEST_IPA
        assert stage2.processor.mem[descriptor, 8] == 0xBADBAD

    def test_stage2_relocates_the_guest_underneath_it(self, stage2):
        stage2.enable_stage2()
        descriptor = stage2.processor.translate_address(stage2.GUEST_VA)
        assert descriptor.paddress.physicaladdress == stage2.REAL_PA
        assert stage2.processor.mem[descriptor, 8] == 0xC0FFEE

    def test_the_guest_tables_are_unchanged(self, stage2):
        # The guest still believes it is using 0x200000; nothing in its own tables
        # records the relocation.
        stage2.enable_stage2()
        stage2.processor.translate_address(stage2.GUEST_VA)
        assert stage2.processor.registers.get_system_register(*TTBR0_EL1) == stage2.stage1_root

    def test_offsets_survive_both_stages(self, stage2):
        stage2.enable_stage2()
        descriptor = stage2.processor.translate_address(stage2.GUEST_VA + 0x123)
        assert descriptor.paddress.physicaladdress == stage2.REAL_PA + 0x123

    def test_stage2_can_take_away_write_permission(self, stage2):
        # The guest's own tables say writable; stage 2 overrules them.
        stage2.enable_stage2()
        stage2.set_stage2_permissions(S2_READ)
        stage2.processor.translate_address(stage2.GUEST_VA)
        with pytest.raises(DataAbortException) as caught:
            stage2.processor.translate_address(stage2.GUEST_VA, is_write=True)
        assert caught.value.status >> 2 == 0b0011      # permission fault

    def test_an_unmapped_intermediate_address_faults(self, stage2):
        stage2.enable_stage2()
        stage2.unmap_stage2()
        with pytest.raises(DataAbortException) as caught:
            stage2.processor.translate_address(stage2.GUEST_VA)
        assert caught.value.status >> 2 == 0b0001      # translation fault

    def test_a_stage2_fault_is_routed_to_the_hypervisor(self, stage2):
        stage2.enable_stage2()
        stage2.unmap_stage2()
        with pytest.raises(DataAbortException) as caught:
            stage2.processor.translate_address(stage2.GUEST_VA)
        assert caught.value.from_stage2 is True

    def test_the_fault_registers_describe_both_addresses(self, stage2):
        stage2.enable_stage2()
        stage2.set_stage2_permissions(S2_READ)
        with pytest.raises(DataAbortException):
            stage2.processor.translate_address(stage2.GUEST_VA, is_write=True)
        registers = stage2.processor.registers
        # FAR_EL2 holds the guest's virtual address...
        assert registers.far[EL.EL2] == stage2.GUEST_VA
        # ...and HPFAR_EL2 the intermediate one it was trying to reach.
        hpfar = registers.get_system_register(*HPFAR_EL2)
        assert hpfar << 8 == stage2.GUEST_IPA

    def test_stage2_does_not_apply_at_el2(self, stage2):
        # The hypervisor's own accesses are not translated by its guest tables.
        stage2.enable_stage2()
        stage2.processor.registers.pstate.el = EL.EL2
        stage2.processor.mmu.flush()
        # EL2 has no stage 1 tables configured, so translation is flat there.
        descriptor = stage2.processor.translate_address(0x900000)
        assert descriptor.paddress.physicaladdress == 0x900000
