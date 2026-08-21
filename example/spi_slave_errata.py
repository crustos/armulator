"""
Testing a driver against the BCM2835 SPI slave block's real behaviour.

Run with:  python3 example/spi_slave_errata.py

The BCM2835 SPI/BSC slave is one of the least forgiving peripherals on the
Pi: the datasheet chapter is thin, parts of it are wrong, and the block does
not behave like a conventional SPI slave at all.  Each scenario below is a
mistake that looks correct in code review and fails on hardware -- which is
exactly what a test harness should catch before the hardware does.

Sources for the behaviours modelled here are community reverse engineering
of real silicon, not the datasheet; see armulator/peripherals/spi_slave.py
for what is confirmed and what is uncertain.
"""

from armulator.boards import RaspberryPi3, RaspberryPi4
from armulator.boards.interconnect import Machine, SpiBridge
from armulator.peripherals.spi_slave import (
    CR_BRK, CR_EN, CR_RXE, CR_SPI, CR_TXE, MISO_IDLE, RSR_OE, SLV_CR, SLV_RSR,
    SLV_SLV, Bcm2835SpiSlave, address_octet,
)

ADDRESS = 0x2A


def fresh_slave(**kwargs):
    slave = Bcm2835SpiSlave(**kwargs)
    slave.write_register(SLV_SLV, ADDRESS)
    slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE)
    return slave


def dialogue(slave, payload, read=False, address=ADDRESS):
    slave.select()
    slave.transfer(address_octet(address, read=read))
    out = [slave.transfer(b) for b in payload]
    slave.deselect()
    return bytes(out)


# ----------------------------------------------------------------------
def errata_1_missing_address_octet():
    """
    A master that just shifts data gets nowhere.

    Every dialogue must open with an address/direction octet.  Omit it and
    the first data byte is silently consumed as the header instead -- so the
    payload is short by one and may be routed to the wrong direction.
    """
    correct = fresh_slave()
    dialogue(correct, b'\xDE\xAD')

    naive = fresh_slave()
    naive.select()
    for byte in b'\xDE\xAD':          # no header: 0xDE becomes the address
        naive.transfer(byte)
    naive.deselect()

    print('1. missing address octet')
    print(f'   with header    -> RX FIFO = {correct.received.hex()}')
    print(f'   without header -> RX FIFO = {naive.received.hex() or "(empty)"}')
    assert correct.received == b'\xDE\xAD'
    assert naive.received == b''      # 0xDE has LSB 0, addressed 0x6F: not us


def errata_2_half_duplex():
    """
    You cannot read and write in the same transaction.

    Conventional SPI slaves shift out while shifting in.  This block does
    not: during a write dialogue MISO idles high and the TX FIFO is never
    touched.  A driver expecting a status byte back while sending a command
    will read 0xFF forever.
    """
    slave = fresh_slave()
    slave.queue_transmit(b'\x99')     # driver thinks this will come back
    returned = dialogue(slave, b'\x01\x02')

    print('\n2. half duplex')
    print(f'   MISO during write  = {returned.hex()}')
    print(f'   TX FIFO afterwards = {slave.pending_transmit.hex()} (untouched)')
    assert returned == bytes([MISO_IDLE, MISO_IDLE])
    assert slave.pending_transmit == b'\x99'


def errata_3_brk_does_not_clear_fifos():
    """
    CR.BRK is documented as clearing the FIFOs.  It does not.

    Firmware that sets BRK during init to discard stale TX data will
    transmit that stale data on the next read dialogue.  Compare the two
    models: one reproduces hardware, one reproduces the datasheet.
    """
    real = fresh_slave()                                  # hardware behaviour
    real.queue_transmit(b'\xDE\xAD')
    real.write_register(SLV_CR, CR_EN | CR_SPI | CR_TXE | CR_RXE | CR_BRK)
    stale = dialogue(real, b'\x00\x00', read=True)

    per_datasheet = fresh_slave(brk_clears_fifos=True)
    per_datasheet.queue_transmit(b'\xDE\xAD')
    per_datasheet.write_register(
        SLV_CR, CR_EN | CR_SPI | CR_TXE | CR_RXE | CR_BRK
    )
    clean = dialogue(per_datasheet, b'\x00\x00', read=True)

    print('\n3. BRK does not clear the FIFOs')
    print(f'   real hardware  -> master reads {stale.hex()} (stale data leaks)')
    print(f'   per datasheet  -> master reads {clean.hex()}')
    assert stale == b'\xDE\xAD'
    assert clean == bytes([MISO_IDLE, MISO_IDLE])


def errata_4_rx_overrun_is_silent():
    """
    The RX FIFO drops bytes when full and only reports it in RSR.

    A driver that does not check RSR.OE will lose data with no other
    indication -- the dialogue completes normally.
    """
    slave = fresh_slave()
    sent = bytes(range(slave.fifo_depth + 4))
    dialogue(slave, sent)

    print('\n4. silent RX overrun')
    print(f'   sent {len(sent)} bytes into a {slave.fifo_depth}-byte FIFO')
    print(f'   kept {len(slave.received)}, RSR.OE = '
          f'{bool(slave.read_register(SLV_RSR) & RSR_OE)}')
    assert len(slave.received) == slave.fifo_depth
    assert slave.read_register(SLV_RSR) & RSR_OE


def errata_5_startup_ordering():
    """
    The slave must be enabled before the master starts talking.

    Under any scheduler -- and on a real bench -- a master that begins its
    dialogue before the slave's CR is written loses the entire transaction,
    with no error anywhere.
    """
    def run(configure_slave_first):
        master, slave = RaspberryPi4(), RaspberryPi3()
        if configure_slave_first:
            slave.spi_slave.write_register(SLV_SLV, ADDRESS)
            slave.spi_slave.write_register(
                SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE
            )
        machine = Machine()
        machine.add('master', master)
        machine.add('slave', slave)
        SpiBridge(master, slave)
        master.spi.write_register(0x00, 0x80)
        master.spi.write_register(0x04, address_octet(ADDRESS, read=False))
        master.spi.write_register(0x04, 0x42)
        master.spi.write_register(0x00, 0x00)
        return slave.spi_slave.received

    late = run(configure_slave_first=False)
    early = run(configure_slave_first=True)

    print('\n5. startup ordering')
    print(f'   slave enabled late  -> RX = {late.hex() or "(empty, silently lost)"}')
    print(f'   slave enabled first -> RX = {early.hex()}')
    assert late == b''
    assert early == b'\x42'


if __name__ == '__main__':
    errata_1_missing_address_octet()
    errata_2_half_duplex()
    errata_3_brk_does_not_clear_fifos()
    errata_4_rx_overrun_is_silent()
    errata_5_startup_ordering()
    print('\nAll errata reproduced.')
