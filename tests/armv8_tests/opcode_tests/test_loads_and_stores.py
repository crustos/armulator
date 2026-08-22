"""
Loads and stores. Snippets set up a base pointer into RAM, store, then read back.
"""

BUF = 0x8000


class TestBasicAccess:
    def test_store_and_load_all_widths(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movn x0, #0\n str x0, [x1]\n'
                       ' ldr x2, [x1]\n ldr w3, [x1]\n ldrb w4, [x1]\n ldrh w5, [x1]')
        assert proc.registers.get_x(2) == 0xFFFFFFFFFFFFFFFF
        assert proc.registers.get_x(3) == 0xFFFFFFFF
        assert proc.registers.get_x(4) == 0xFF
        assert proc.registers.get_x(5) == 0xFFFF

    def test_32_bit_load_zeroes_the_upper_half(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movn x0, #0\n str x0, [x1]\n ldr w2, [x1]')
        assert proc.registers.get_x(2) == 0x00000000FFFFFFFF

    def test_sign_extending_load(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movz x0, #0x80\n strb w0, [x1]\n'
                       ' ldrsb x2, [x1]\n ldrb w3, [x1]')
        assert proc.registers.get_x(2) == 0xFFFFFFFFFFFFFF80
        assert proc.registers.get_x(3) == 0x80

    def test_byte_store_does_not_disturb_neighbours(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movn x0, #0\n str x0, [x1]\n'
                       ' movz w2, #0\n strb w2, [x1]\n ldr x3, [x1]')
        assert proc.registers.get_x(3) == 0xFFFFFFFFFFFFFF00


class TestAddressingModes:
    def test_unsigned_offset_is_scaled_by_access_width(self, run_a64):
        # #8 on a doubleword access means eight bytes, not eight elements.
        proc = run_a64(f'movz x1, #{BUF}\n movz x0, #0xAB\n str x0, [x1, #8]\n'
                       ' ldr x2, [x1, #8]\n ldr x3, [x1]')
        assert proc.registers.get_x(2) == 0xAB
        assert proc.registers.get_x(3) == 0

    def test_unscaled_negative_offset(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movz x0, #0x5A\n stur x0, [x1, #-8]\n'
                       ' ldur x2, [x1, #-8]')
        assert proc.registers.get_x(2) == 0x5A

    def test_pre_index_writes_back_before_the_access(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movz x0, #1\n str x0, [x1, #8]!\n'
                       f' movz x3, #{BUF}\n ldr x2, [x3, #8]')
        assert proc.registers.get_x(1) == BUF + 8
        assert proc.registers.get_x(2) == 1

    def test_post_index_writes_back_after_the_access(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movz x0, #1\n str x0, [x1], #8\n'
                       f' movz x3, #{BUF}\n ldr x2, [x3]')
        assert proc.registers.get_x(1) == BUF + 8
        assert proc.registers.get_x(2) == 1

    def test_register_offset_with_scaling(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movz x2, #2\n movz x0, #0x55\n'
                       ' str x0, [x1, x2, lsl #3]\n ldr x3, [x1, #16]')
        assert proc.registers.get_x(3) == 0x55

    def test_register_offset_sign_extends_a_word_index(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movn w2, #0\n movz x0, #0x77\n'
                       ' str x0, [x1, w2, sxtw #3]\n ldr x3, [x1, #-8]')
        assert proc.registers.get_x(3) == 0x77


class TestPair:
    def test_store_and_load_pair(self, run_a64):
        proc = run_a64(f'movz x2, #{BUF}\n movz x0, #0x1111\n movz x1, #0x2222\n'
                       ' stp x0, x1, [x2]\n ldp x3, x4, [x2]')
        assert proc.registers.get_x(3) == 0x1111
        assert proc.registers.get_x(4) == 0x2222

    def test_pair_offset_is_scaled(self, run_a64):
        proc = run_a64(f'movz x2, #{BUF}\n movz x0, #7\n movz x1, #8\n'
                       ' stp x0, x1, [x2, #16]\n ldp x3, x4, [x2, #16]')
        assert (proc.registers.get_x(3), proc.registers.get_x(4)) == (7, 8)

    def test_frame_push_and_pop(self, run_a64):
        # The standard prologue and epilogue: SP must come back where it started.
        proc = run_a64(f'movz x0, #{BUF}\n mov sp, x0\n movz x29, #0xAAA\n'
                       ' movz x30, #0xBBB\n stp x29, x30, [sp, #-16]!\n'
                       ' movz x29, #0\n movz x30, #0\n ldp x29, x30, [sp], #16')
        assert proc.registers.get_x(29) == 0xAAA
        assert proc.registers.get_x(30) == 0xBBB
        assert proc.registers.get_sp() == BUF

    def test_32_bit_pair_uses_a_four_byte_stride(self, run_a64):
        proc = run_a64(f'movz x2, #{BUF}\n movz w0, #1\n movz w1, #2\n'
                       ' stp w0, w1, [x2]\n ldr x3, [x2]')
        # Two words packed little endian into one doubleword.
        assert proc.registers.get_x(3) == 0x0000000200000001


class TestLiteral:
    def test_load_literal_reads_the_constant_pool(self, run_a64):
        proc = run_a64('ldr x0, pool\n b done\n .balign 8\npool: .quad 0x1122334455667788\ndone: nop',
                       steps=2)
        assert proc.registers.get_x(0) == 0x1122334455667788


class TestExclusive:
    def test_exclusive_store_succeeds_after_a_reservation(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movz x0, #0x42\n ldxr x4, [x1]\n'
                       ' stxr w2, x0, [x1]\n ldxr x3, [x1]')
        assert proc.registers.get_x(2) == 0
        assert proc.registers.get_x(3) == 0x42

    def test_exclusive_store_without_a_reservation_fails(self, run_a64):
        # No LDXR, so there is nothing to store exclusively against.
        proc = run_a64(f'movz x1, #{BUF}\n movz x0, #0x42\n stxr w2, x0, [x1]\n'
                       ' ldr x3, [x1]')
        assert proc.registers.get_x(2) == 1
        assert proc.registers.get_x(3) == 0     # the store must not have happened

    def test_clrex_drops_the_reservation(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movz x0, #0x42\n ldxr x4, [x1]\n'
                       ' clrex\n stxr w2, x0, [x1]')
        assert proc.registers.get_x(2) == 1

    def test_acquire_release_pair(self, run_a64):
        proc = run_a64(f'movz x1, #{BUF}\n movz x0, #0x99\n stlr x0, [x1]\n ldar x2, [x1]')
        assert proc.registers.get_x(2) == 0x99


class TestStackPointerBase:
    def test_sp_relative_access(self, run_a64):
        proc = run_a64(f'movz x0, #{BUF}\n mov sp, x0\n movz x1, #0xCD\n'
                       ' str x1, [sp, #8]\n ldr x2, [sp, #8]')
        assert proc.registers.get_x(2) == 0xCD
