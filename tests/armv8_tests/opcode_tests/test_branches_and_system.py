"""
Branches, exception generation and the system instruction space.
"""

from armulator.armv8.enums import EL


class TestUnconditionalBranch:
    def test_branch_forward(self, run_a64):
        proc = run_a64('b target\n movz x0, #1\ntarget: movz x1, #2')
        assert proc.registers.get_x(0) == 0
        assert proc.registers.get_x(1) == 2

    def test_branch_backward_loops(self, run_a64):
        proc = run_a64('movz x0, #0\n movz x1, #5\nloop: add x0, x0, x1\n'
                       ' sub x1, x1, #1\n cbnz x1, loop', steps=30)
        assert proc.registers.get_x(0) == 15

    def test_bl_sets_the_link_register(self, run_a64):
        proc, _ = run_a64.build('bl func\n nop\nfunc: nop')
        proc.emulate_cycle()
        # The return address is the instruction after the call.
        assert proc.registers.get_lr() == 0x1004
        assert proc.registers.get_pc() == 0x1008

    def test_bl_and_ret_round_trip(self, run_a64):
        proc = run_a64('movz x0, #0\n bl func\n movz x2, #7\n b end\n'
                       'func: movz x0, #99\n ret\nend: nop')
        assert proc.registers.get_x(0) == 99
        assert proc.registers.get_x(2) == 7


class TestConditionalBranch:
    def test_taken_when_condition_holds(self, run_a64):
        proc = run_a64('movz x0, #5\n cmp x0, #5\n b.eq taken\n'
                       ' movz x1, #111\n b done\ntaken: movz x1, #222\ndone: nop')
        assert proc.registers.get_x(1) == 222

    def test_not_taken_when_condition_fails(self, run_a64):
        proc = run_a64('movz x0, #5\n cmp x0, #4\n b.eq taken\n'
                       ' movz x1, #111\n b done\ntaken: movz x1, #222\ndone: nop')
        assert proc.registers.get_x(1) == 111


class TestCompareAndBranch:
    def test_cbz_branches_on_zero(self, run_a64):
        proc = run_a64('movz x0, #0\n cbz x0, taken\n movz x1, #1\n b d\n'
                       'taken: movz x1, #2\nd: nop')
        assert proc.registers.get_x(1) == 2

    def test_cbnz_does_not_branch_on_zero(self, run_a64):
        proc = run_a64('movz x0, #0\n cbnz x0, taken\n movz x1, #1\n b d\n'
                       'taken: movz x1, #2\nd: nop')
        assert proc.registers.get_x(1) == 1

    def test_cbz_uses_only_the_low_word_in_32_bit_form(self, run_a64):
        # The upper half is set, but the 32-bit form must ignore it.
        proc = run_a64('movz x0, #1, lsl #32\n cbz w0, taken\n movz x1, #1\n b d\n'
                       'taken: movz x1, #2\nd: nop')
        assert proc.registers.get_x(1) == 2


class TestTestAndBranch:
    def test_tbnz_on_a_set_bit(self, run_a64):
        proc = run_a64('movz x0, #0x100\n tbnz x0, #8, hit\n movz x1, #0\n b d\n'
                       'hit: movz x1, #1\nd: nop')
        assert proc.registers.get_x(1) == 1

    def test_tbz_reaches_bits_above_31(self, run_a64):
        # The bit number is split across the encoding; bit 40 exercises the top half.
        proc = run_a64('movz x0, #1, lsl #48\n tbz x0, #40, hit\n movz x1, #0\n b d\n'
                       'hit: movz x1, #1\nd: nop')
        assert proc.registers.get_x(1) == 1


class TestBranchRegister:
    def test_br_jumps_to_a_computed_address(self, run_a64):
        proc = run_a64('adr x0, target\n br x0\n movz x1, #1\ntarget: movz x1, #2')
        assert proc.registers.get_x(1) == 2

    def test_blr_links(self, run_a64):
        proc = run_a64('adr x0, func\n blr x0\n b end\nfunc: movz x1, #5\n ret\nend: nop')
        assert proc.registers.get_x(1) == 5


class TestHintsAndBarriers:
    def test_nop_and_barriers_retire(self, run_a64):
        proc = run_a64('nop\n dsb sy\n dmb ish\n isb\n movz x0, #1')
        assert proc.registers.get_x(0) == 1
        assert proc.registers.get_pc() == 0x1014

    def test_wfi_parks_the_processor(self, run_a64):
        proc = run_a64('wfi')
        assert proc.is_wait_for_interrupt is True

    def test_sev_sets_the_event_register(self, run_a64):
        proc = run_a64('sev')
        assert proc.event_registered() is True


class TestSystemRegisters:
    def test_mrs_reads_midr(self, run_a64):
        proc = run_a64('mrs x0, midr_el1')
        assert proc.registers.get_x(0) == 0x411FD070

    def test_msr_mrs_round_trip(self, run_a64):
        proc = run_a64('movz x1, #0x1234\n msr vbar_el1, x1\n mrs x2, vbar_el1')
        assert proc.registers.get_x(2) == 0x1234
        assert proc.registers.vbar[EL.EL1] == 0x1234

    def test_daifclr_unmasks_irq(self, run_a64):
        proc = run_a64('msr daifclr, #2')
        assert proc.registers.pstate.i == 0

    def test_daifset_masks_irq(self, run_a64):
        proc = run_a64('msr daifclr, #2\n msr daifset, #2')
        assert proc.registers.pstate.i == 1

    def test_cache_maintenance_retires_without_faulting(self, run_a64):
        # There are no caches to maintain, but startup code issues these freely.
        proc = run_a64('ic iallu\n dc civac, x0\n tlbi vmalle1\n movz x0, #1')
        assert proc.registers.get_x(0) == 1


class TestExceptionGeneration:
    def _with_vectors(self, run_a64, source, steps):
        proc, _ = run_a64.build(source)
        proc.registers.vbar[EL.EL1] = 0x2000
        for _ in range(steps):
            proc.emulate_cycle()
        return proc

    def test_svc_vectors_and_records_its_immediate(self, run_a64):
        proc = self._with_vectors(run_a64, 'movz x0, #1\n svc #7', 2)
        assert proc.registers.get_pc() == 0x2200
        assert proc.registers.esr[EL.EL1] >> 26 == 0x15
        assert proc.registers.esr[EL.EL1] & 0xFFFF == 7

    def test_svc_returns_to_the_following_instruction(self, run_a64):
        proc = self._with_vectors(run_a64, 'movz x0, #1\n svc #0\n movz x0, #2', 2)
        # SVC completes, so the saved return address is the next instruction.
        assert proc.registers.elr[EL.EL1] == 0x1008

    def test_brk_reports_a_breakpoint(self, run_a64):
        proc = self._with_vectors(run_a64, 'brk #3', 1)
        assert proc.registers.esr[EL.EL1] >> 26 == 0x3C
