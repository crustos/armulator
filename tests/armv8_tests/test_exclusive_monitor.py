"""
The exclusive monitor on its own, without the surrounding processor.
"""

from armulator.armv8.exclusive_monitor import ExclusiveMonitor


class TestReservations:
    def test_store_succeeds_after_a_reservation(self):
        monitor = ExclusiveMonitor()
        monitor.reserve(0, 0x1000)
        assert monitor.check_and_clear(0, 0x1000) is True

    def test_store_without_a_reservation_fails(self):
        monitor = ExclusiveMonitor()
        assert monitor.check_and_clear(0, 0x1000) is False

    def test_a_reservation_is_consumed_by_a_successful_store(self):
        monitor = ExclusiveMonitor()
        monitor.reserve(0, 0x1000)
        assert monitor.check_and_clear(0, 0x1000) is True
        # A second store has nothing left to go on.
        assert monitor.check_and_clear(0, 0x1000) is False

    def test_clrex_drops_the_reservation(self):
        monitor = ExclusiveMonitor()
        monitor.reserve(0, 0x1000)
        monitor.clear(0)
        assert monitor.check_and_clear(0, 0x1000) is False


class TestContention:
    def test_only_one_core_wins(self):
        monitor = ExclusiveMonitor()
        monitor.reserve(0, 0x1000)
        monitor.reserve(1, 0x1000)
        assert monitor.check_and_clear(0, 0x1000) is True
        # Core 0's success cleared core 1's reservation, so core 1 must retry.
        assert monitor.check_and_clear(1, 0x1000) is False

    def test_reservations_on_different_blocks_do_not_interfere(self):
        monitor = ExclusiveMonitor()
        monitor.reserve(0, 0x1000)
        monitor.reserve(1, 0x2000)
        assert monitor.check_and_clear(0, 0x1000) is True
        assert monitor.check_and_clear(1, 0x2000) is True

    def test_a_plain_store_breaks_another_cores_reservation(self):
        # This is what stops a lock released with STR from leaving a stale reservation
        # that would let a second core "succeed" at acquiring it.
        monitor = ExclusiveMonitor()
        monitor.reserve(1, 0x1000)
        monitor.notify_store(0, 0x1000, 4)
        assert monitor.check_and_clear(1, 0x1000) is False

    def test_a_plain_store_keeps_the_storing_cores_own_reservation(self):
        monitor = ExclusiveMonitor()
        monitor.reserve(0, 0x1000)
        monitor.notify_store(0, 0x1000, 4)
        assert monitor.check_and_clear(0, 0x1000) is True


class TestGranule:
    def test_addresses_in_the_same_block_share_a_reservation(self):
        # The reservation granule is 16 bytes on a Cortex-A57, so nearby variables can
        # make each other's exclusive stores fail.
        monitor = ExclusiveMonitor()
        monitor.reserve(0, 0x1000)
        assert monitor.check_and_clear(0, 0x1008) is True

    def test_addresses_in_different_blocks_do_not(self):
        monitor = ExclusiveMonitor()
        monitor.reserve(0, 0x1000)
        assert monitor.check_and_clear(0, 0x1010) is False
