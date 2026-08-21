"""
Two boards talking to each other: a Raspberry Pi 4 and a Jetson Nano.

Run with:  python3 example/two_device_link.py

This is the arrangement you would build on a bench -- a Pi wired to a Jetson
over a few GPIO lines and an SPI bus -- except both sides are emulated and
both sides run real ARM firmware.

    Pi 4                          Jetson Nano
    ----                          -----------
    GPIO17  --------- DATA ------> PA0
    GPIO27  <-------- READY ------ PA1
    SPI0 (master)  --- MOSI/MISO --> SPI (slave)

Scenarios below, in order:

  1. one-way signal      Pi raises a line, Jetson observes it
  2. handshake           Jetson answers on a second line, Pi waits for it
  3. SPI payload         Pi 4 master shifts bytes to a Pi 3 SPI slave
  4. interrupt routing   the incoming line raises a real GIC interrupt
"""

from armulator.boards import JetsonNano, RaspberryPi3, RaspberryPi4
from armulator.boards.firmware import firmware
from armulator.boards.interconnect import GpioLink, Machine, SpiBridge
from armulator.peripherals.spi_slave import address_octet
from armulator.peripherals.gic400 import (
    GICC_CTLR, GICC_IAR, GICC_PMR, GICD_CTLR, GICD_ICFGR, GICD_ISENABLER,
)

PI_GPIO = 0xFE200000
PI_SPI = 0xFE204000
NANO_GPIO = 0x6000D000

DATA_PIN = 17          # Pi drives, Jetson listens
READY_PIN = 27         # Jetson drives, Pi listens


def build_pi(source, trace=False):
    board = RaspberryPi4(trace=trace)
    board.load(board.CODE_BASE, firmware(source, address=board.CODE_BASE))
    board.start()
    return board


def build_nano(source, trace=False):
    board = JetsonNano(trace=trace)
    board.load(board.CODE_BASE, firmware(source, address=board.CODE_BASE))
    board.start()
    return board


# ----------------------------------------------------------------------
def scenario_1_one_way_signal():
    """The Pi asserts a line; the Jetson sees it on its own pin."""
    pi = build_pi(f"""
        ldr r0, ={PI_GPIO:#x}
        mov r1, #1
        lsl r1, r1, #21            @ GPIO17 -> output
        str r1, [r0, #0x04]
        mov r2, #1
        lsl r2, r2, #17
        str r2, [r0, #0x1C]        @ GPSET0: raise DATA
    """)
    # The Jetson just configures PA0 as a GPIO input and parks.
    nano = build_nano(f"""
        ldr r0, ={NANO_GPIO:#x}
        mov r1, #1
        str r1, [r0, #0x00]        @ CNF: PA0 is GPIO
        mov r1, #0
        str r1, [r0, #0x10]        @ OE: input
    """)

    machine = Machine()
    machine.add('pi', pi)
    machine.add('nano', nano)
    wire = machine.link(GpioLink(pi, DATA_PIN, nano, 'PA0', name='DATA'))
    machine.run(2000)

    print('1. one-way signal')
    print(f'   Pi   GPIO{DATA_PIN} = {pi.gpio.level(DATA_PIN)}')
    print(f'   Nano PA0      = {nano.gpio.level("PA0")}')
    print(f'   wire edges    = {wire.edges()}')
    assert pi.gpio.level(DATA_PIN) is True
    assert nano.gpio.level('PA0') is True


# ----------------------------------------------------------------------
def scenario_2_handshake():
    """
    A real two-way handshake.

    The Pi raises DATA then spins waiting for READY.  The Jetson spins
    waiting for DATA, then raises READY.  Neither can finish alone -- if
    the interconnect were not propagating, both would hang.
    """
    pi = build_pi(f"""
        ldr r0, ={PI_GPIO:#x}
        mov r1, #1
        lsl r1, r1, #21            @ GPIO17 output
        str r1, [r0, #0x04]
        mov r2, #1
        lsl r2, r2, #17
        str r2, [r0, #0x1C]        @ raise DATA
    wait_ready:
        ldr r3, [r0, #0x34]        @ GPLEV0
        tst r3, #(1 << 27)         @ READY from the Jetson?
        beq wait_ready
        mov r4, #1                 @ marker: handshake complete
    """)
    nano = build_nano(f"""
        ldr r0, ={NANO_GPIO:#x}
        mov r1, #3
        str r1, [r0, #0x00]        @ CNF: PA0 and PA1 are GPIO
        mov r1, #2
        str r1, [r0, #0x10]        @ OE: PA1 output, PA0 input
    wait_data:
        ldr r2, [r0, #0x30]        @ IN
        tst r2, #1                 @ DATA asserted?
        beq wait_data
        mov r3, #2
        str r3, [r0, #0x20]        @ OUT: raise READY on PA1
    """)

    machine = Machine(slice_size=8)
    machine.add('pi', pi)
    machine.add('nano', nano)
    machine.link(GpioLink(pi, DATA_PIN, nano, 'PA0', name='DATA'))
    machine.link(GpioLink(nano, 'PA1', pi, READY_PIN, name='READY'))

    completed = machine.run_until(
        lambda: pi.cpu.registers.get(4) == 1, max_instructions=20000
    )

    print('\n2. handshake')
    print(f'   Nano saw DATA, raised READY = {nano.gpio.level("PA1")}')
    print(f'   Pi saw READY, r4            = {pi.cpu.registers.get(4)}')
    print(f'   handshake completed         = {completed}')
    assert completed, 'handshake did not complete'
    assert nano.gpio.level('PA1') is True


# ----------------------------------------------------------------------
SLAVE_ADDRESS = 0x2A


def scenario_3_spi_payload():
    """
    A Pi 4 master shifts a payload to a Pi 3 acting as SPI slave.

    The slave end is a Pi rather than the Jetson deliberately: the BCM2835
    SPI slave is a real modelled peripheral, while the Jetson has no SPI
    slave model at all (its master is now a real Tegra controller, but
    slave mode is not implemented).  Bridging to it is refused, not faked.

    Note the dialogue header.  This block is half duplex and expects an
    address/direction octet first -- a master that simply shifts data is
    ignored, on hardware and here.
    """
    payload = b'\xDE\xAD\xBE\xEF'
    body = f"""
        ldr r0, ={PI_SPI:#x}
        mov r1, #0x80              @ CS: TA=1, chip select 0
        str r1, [r0, #0x00]
        mov r2, #{address_octet(SLAVE_ADDRESS, read=False)}
        str r2, [r0, #0x04]        @ address octet: write to 0x2A
    """
    for byte in payload:
        body += f"""
        mov r2, #{byte}
        str r2, [r0, #0x04]        @ FIFO: shift out 0x{byte:02X}
    """
    body += """
        mov r1, #0
        str r1, [r0, #0x00]        @ TA=0: end the dialogue
    """
    master = build_pi(body, trace=True)

    slave = RaspberryPi3()
    slave.load(slave.CODE_BASE, firmware(f"""
        ldr r0, ={0x3F214000:#x}
        mov r1, #{SLAVE_ADDRESS}
        str r1, [r0, #0x08]        @ SLV: our address
        mov r2, #0x300
        orr r2, r2, #0x3           @ TXE | RXE | SPI | EN
        str r2, [r0, #0x0C]        @ CR
    """, address=slave.CODE_BASE))
    slave.start()
    # Bring the slave up before the master starts talking.  Under the
    # round-robin scheduler the master would otherwise complete its whole
    # dialogue in its first slice, against a block that is not yet enabled
    # -- the same startup-ordering hazard as on a real bench.
    slave.run(500)

    machine = Machine()
    machine.add('master', master)
    machine.add('slave', slave)
    bridge = SpiBridge(master, slave, chip_select=0)
    machine.run(5000)

    print('\n3. SPI payload')
    print(f'   master shifted (with header) = {bridge.mosi.hex()}')
    print(f'   slave RX FIFO                = {slave.spi_slave.received.hex()}')
    print(f'   dialogues seen               = {slave.spi_slave.dialogues}')
    assert slave.spi_slave.received == payload


# ----------------------------------------------------------------------
def scenario_4_interrupt_routing():
    """
    The incoming line raises a genuine GIC interrupt on the Pi.

    The Jetson drives READY; the Pi has GPIO edge detection armed and the
    GIC-400 configured, so the signal crossing the wire produces a real
    acknowledge/EOI cycle rather than a polled flag.
    """
    pi = build_pi(f"""
        ldr r0, ={PI_GPIO:#x}
        mov r1, #(1 << 27)
        str r1, [r0, #0x4C]        @ GPREN0: rising edge on GPIO27
    """)
    nano = build_nano(f"""
        ldr r0, ={NANO_GPIO:#x}
        mov r1, #2
        str r1, [r0, #0x00]        @ CNF: PA1 GPIO
        str r1, [r0, #0x10]        @ OE:  output
        str r1, [r0, #0x20]        @ OUT: raise READY
    """)

    # Configure the Pi's GIC: enable the distributor, the CPU interface,
    # unmask all priorities, and enable the GPIO interrupt line.
    gic = pi.gic
    intid = gic.connect(pi.gpio, RaspberryPi4.GPIO_SPI)
    gic.write_register(GICD_CTLR, 1)
    gic.write_register(GICC_CTLR, 1)
    gic.write_register(GICC_PMR, 0xFF)
    gic.write_register(GICD_ISENABLER + 4 * (intid // 32), 1 << (intid % 32))

    machine = Machine()
    machine.add('pi', pi)
    machine.add('nano', nano)
    machine.link(GpioLink(nano, 'PA1', pi, READY_PIN, name='READY'))
    machine.run(2000)

    gic.refresh()
    acknowledged = gic.read_register(GICC_IAR)
    print('\n4. interrupt routing')
    print(f'   GPIO27 level        = {pi.gpio.level(READY_PIN)}')
    print(f'   GIC interrupt ID    = {intid}')
    print(f'   acknowledged (IAR)  = {acknowledged}')
    assert pi.gpio.level(READY_PIN) is True
    assert acknowledged == intid, f'expected {intid}, got {acknowledged}'


if __name__ == '__main__':
    scenario_1_one_way_signal()
    scenario_2_handshake()
    scenario_3_spi_payload()
    scenario_4_interrupt_routing()
    print('\nAll scenarios passed.')
