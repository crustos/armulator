"""
Generate regression baseline traces from the models.

Run with:  python3 tools/record_baselines.py

IMPORTANT: these traces are recorded from armulator's own models, not from
hardware.  Replaying them detects *regressions* -- a change that alters
peripheral behaviour -- but proves nothing about whether the models match
real silicon.  Every file is stamped with that provenance, and the replay
report repeats the warning.

To validate against hardware you need a real capture; see RASPI.md for the
ftrace recipe.
"""

from pathlib import Path

from armulator.boards import JetsonNano, RaspberryPi3, RaspberryPi4
from armulator.boards.firmware import firmware
from armulator.harness import TraceRecorder
from armulator.peripherals.spi_slave import (
    CR_EN, CR_RXE, CR_SPI, CR_TXE, SLV_CR, SLV_SLV, address_octet,
)

TRACES = Path(__file__).resolve().parent.parent / 'traces'
SLAVE_ADDRESS = 0x2A


def run(board, source, budget=5000):
    board.load(board.CODE_BASE, firmware(source, address=board.CODE_BASE))
    board.start()
    board.run(budget)
    return board


def gpio_blink():
    """A GPIO output configured and toggled several times."""
    board = RaspberryPi4()
    recorder = TraceRecorder(board.gpio, 0xFE200000, name='gpio_blink')
    run(board, """
        ldr r0, =0xFE200000
        mov r1, #1
        lsl r1, r1, #21
        str r1, [r0, #0x04]
        mov r2, #1
        lsl r2, r2, #17
        mov r3, #3
    loop:
        str r2, [r0, #0x1C]
        str r2, [r0, #0x28]
        ldr r4, [r0, #0x34]
        subs r3, r3, #1
        bne loop
    """)
    return recorder.trace()


def gpio_pull_bcm2711():
    """The Pi 4's direct pull control path."""
    board = RaspberryPi4()
    recorder = TraceRecorder(board.gpio, 0xFE200000, name='gpio_pull_bcm2711')
    run(board, """
        ldr r0, =0xFE200000
        mov r1, #1
        lsl r1, r1, #8
        str r1, [r0, #0xE4]
        ldr r2, [r0, #0xE4]
    """)
    return recorder.trace()


def uart_hello():
    """UART init and a short transmission."""
    board = RaspberryPi4()
    recorder = TraceRecorder(board.uart, 0xFE201000, name='uart_hello')
    run(board, """
        ldr r0, =0xFE201000
        mov r1, #26
        str r1, [r0, #0x24]
        mov r1, #3
        str r1, [r0, #0x28]
        mov r1, #0x70
        str r1, [r0, #0x2C]
        mov r2, #0x48
        str r2, [r0, #0x00]
        ldr r3, [r0, #0x18]
        mov r2, #0x49
        str r2, [r0, #0x00]
    """)
    return recorder.trace()


def spi_master_transfer():
    """SPI master opening a dialogue and shifting bytes."""
    board = RaspberryPi4()
    recorder = TraceRecorder(board.spi, 0xFE204000, name='spi_master_transfer')
    run(board, f"""
        ldr r0, =0xFE204000
        mov r1, #250
        str r1, [r0, #0x08]
        mov r1, #0x80
        str r1, [r0, #0x00]
        mov r2, #{address_octet(SLAVE_ADDRESS, read=False)}
        str r2, [r0, #0x04]
        mov r2, #0x42
        str r2, [r0, #0x04]
        ldr r3, [r0, #0x00]
        mov r1, #0
        str r1, [r0, #0x00]
    """)
    return recorder.trace()


def i2c_write():
    """An I2C register write transaction."""
    board = RaspberryPi4()
    from armulator.peripherals.serial_bus import I2cSlaveDevice
    board.i2c.attach_slave(I2cSlaveDevice(address=0x48))
    recorder = TraceRecorder(board.i2c, 0xFE804000, name='i2c_write')
    run(board, """
        ldr r0, =0xFE804000
        mov r1, #0x48
        str r1, [r0, #0x0C]
        mov r2, #0x01
        str r2, [r0, #0x10]
        mov r2, #0xEE
        str r2, [r0, #0x10]
        mov r3, #2
        str r3, [r0, #0x08]
        mov r4, #0x8080
        str r4, [r0, #0x00]
        ldr r5, [r0, #0x04]
    """)
    return recorder.trace()


def spi_slave_dialogue():
    """The slave block being configured and receiving a write dialogue."""
    board = RaspberryPi3()
    board.spi_slave.write_register(SLV_SLV, SLAVE_ADDRESS)
    board.spi_slave.write_register(SLV_CR, CR_EN | CR_SPI | CR_RXE | CR_TXE)
    recorder = TraceRecorder(board.spi_slave, 0x3F214000,
                             name='spi_slave_dialogue')
    run(board, """
        ldr r0, =0x3F214000
        ldr r1, [r0, #0x10]
        ldr r2, [r0, #0x1C]
        ldr r3, [r0, #0x08]
    """)
    return recorder.trace()


def tegra_spi_transfer():
    """Tegra SPI master configured and triggered via PIO."""
    from armulator.peripherals.spi_tegra import (
        CMD1_M_S, CMD1_PIO, CMD1_RX_EN, CMD1_TX_EN,
    )
    board = JetsonNano()
    recorder = TraceRecorder(board.spi, 0x7000D400, name='tegra_spi_transfer')
    cmd = CMD1_M_S | CMD1_TX_EN | CMD1_RX_EN | 7
    run(board, f"""
        ldr r0, =0x7000D400
        ldr r1, ={cmd:#x}
        str r1, [r0, #0x000]       @ COMMAND1
        mov r2, #0x54
        str r2, [r0, #0x108]       @ TX_FIFO
        mov r2, #0x42
        str r2, [r0, #0x108]
        mov r3, #1
        str r3, [r0, #0x024]       @ DMA_BLK: two packets
        ldr r1, ={cmd | CMD1_PIO:#x}
        str r1, [r0, #0x000]       @ COMMAND1 with PIO: go
        ldr r4, [r0, #0x010]       @ TRANS_STATUS
        ldr r5, [r0, #0x014]       @ FIFO_STATUS
    """)
    return recorder.trace()


GENERATORS = [
    gpio_blink, gpio_pull_bcm2711, uart_hello,
    spi_master_transfer, i2c_write, spi_slave_dialogue,
    tegra_spi_transfer,
]


def main():
    TRACES.mkdir(exist_ok=True)
    for generator in GENERATORS:
        trace = generator()
        path = TRACES / f'{trace.name}.trace'
        path.write_text(trace.to_canonical())
        print(f'{path.name:<28} {len(trace):>3} accesses')
    readme = TRACES / 'README.md'
    readme.write_text(
        '# Trace files\n\n'
        'Every trace here was **recorded from armulator\'s own models**, not '
        'from hardware. They exist to catch regressions: if a change alters '
        'peripheral behaviour, replaying these will diverge.\n\n'
        'They do **not** validate the models against real silicon. A replay '
        'of a self-recorded trace is circular by construction, and the replay '
        'report says so in its output.\n\n'
        'To validate against hardware, capture a real trace on a Pi and '
        'replay that instead. See `RASPI.md` for the ftrace recipe.\n\n'
        'Regenerate with `python3 tools/record_baselines.py`.\n'
    )
    print(f'\nWrote {len(GENERATORS)} baselines to {TRACES}/')


if __name__ == '__main__':
    main()
