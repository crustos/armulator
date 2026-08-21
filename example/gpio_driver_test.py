"""
Testing a GPIO driver against an emulated Raspberry Pi 4.

Run with:  python3 example/gpio_driver_test.py

Demonstrates the three things the peripheral models are built for:

  1. asserting on *pin state* -- did the driver actually drive the line?
  2. asserting on the *register sequence* -- did it do so correctly?
  3. asserting on *timing-independent protocol shape* -- bit-banged output.
"""

from armulator.boards import RaspberryPi4
from armulator.boards.firmware import firmware
from armulator.peripherals.gpio_bcm import GpioFunction

GPIO_BASE = 0xFE200000


def build(source):
    board = RaspberryPi4(trace=True)
    board.load(board.CODE_BASE, firmware(source, address=board.CODE_BASE))
    board.start()
    board.run(5000)
    return board


def example_1_pin_state():
    """A driver configures pin 17 as an output and asserts it."""
    board = build(f"""
        ldr r0, ={GPIO_BASE:#x}
        mov r1, #1
        lsl r1, r1, #21            @ FSEL17 = output
        str r1, [r0, #0x04]
        mov r2, #1
        lsl r2, r2, #17
        str r2, [r0, #0x1C]        @ GPSET0
    """)
    print('1. pin state')
    print(f'   function(17) = {board.gpio.function(17).name}')
    print(f'   level(17)    = {board.gpio.level(17)}')
    assert board.gpio.function(17) == GpioFunction.OUTPUT
    assert board.gpio.level(17) is True


def example_2_register_sequence():
    """
    The Pi 4 dropped the GPPUD clocked handshake in favour of direct
    control registers.  A driver ported from the Pi 3 that still pokes
    GPPUD will appear to work -- the write goes nowhere -- until you
    check the pull actually took effect.
    """
    legacy = build(f"""
        ldr r0, ={GPIO_BASE:#x}
        mov r1, #2
        str r1, [r0, #0x94]        @ GPPUD = pull-up (Pi 3 style)
        mov r2, #1
        lsl r2, r2, #4
        str r2, [r0, #0x98]        @ GPPUDCLK0, pin 4
    """)
    correct = build(f"""
        ldr r0, ={GPIO_BASE:#x}
        mov r1, #1
        lsl r1, r1, #8             @ pin 4 -> bits 9:8, 01 = pull-up
        str r1, [r0, #0xE4]        @ GPIO_PUP_PDN_CNTRL_REG0
    """)
    print('\n2. register sequence')
    print(f'   Pi 3 style on a Pi 4 -> pull(4) = {legacy.gpio.pull(4).name}')
    print(f'   Pi 4 style           -> pull(4) = {correct.gpio.pull(4).name}')
    assert legacy.gpio.pull(4).name == 'OFF'      # silently did nothing
    assert correct.gpio.pull(4).name == 'UP'


def example_3_bit_banged_protocol():
    """Shift eight bits out MSB-first and recover them from the waveform."""
    byte = 0xA5
    board = build(f"""
        ldr r0, ={GPIO_BASE:#x}
        mov r1, #1
        lsl r1, r1, #24            @ pin 18 (data) output, FSEL8 of GPFSEL1
        str r1, [r0, #0x04]
        mov r4, #{byte}
        mov r5, #8                 @ bit counter
        mov r2, #1
        lsl r2, r2, #18
    next_bit:
        tst r4, #0x80
        strne r2, [r0, #0x1C]      @ GPSET0 if bit set
        streq r2, [r0, #0x28]      @ GPCLR0 otherwise
        lsl r4, r4, #1
        subs r5, r5, #1
        bne next_bit
    """)
    # Reconstruct the transmitted byte from the recorded pin transitions,
    # holding the level between edges the way a receiver would sample it.
    bits, level = [], False
    for access in board.gpio.accesses:
        if access.name == 'GPSET0' and access.value & (1 << 18):
            level = True
            bits.append(1)
        elif access.name == 'GPCLR0' and access.value & (1 << 18):
            level = False
            bits.append(0)
    recovered = int(''.join(str(b) for b in bits), 2)
    print('\n3. bit-banged protocol')
    print(f'   sent      0x{byte:02X} = {byte:08b}')
    print(f'   recovered 0x{recovered:02X} = {recovered:08b}')
    assert recovered == byte


def example_4_input_and_edges():
    """Firmware waits for a rising edge on pin 7, then echoes it to pin 8."""
    board = RaspberryPi4(trace=True)
    board.load(board.CODE_BASE, firmware(f"""
        ldr r0, ={GPIO_BASE:#x}
        mov r1, #1
        lsl r1, r1, #24            @ pin 8 output (FSEL8 of GPFSEL0)
        str r1, [r0, #0x00]
    poll:
        ldr r2, [r0, #0x34]        @ GPLEV0
        tst r2, #0x80              @ pin 7 high?
        beq poll
        mov r3, #1
        lsl r3, r3, #8
        str r3, [r0, #0x1C]        @ mirror to pin 8
    """, address=board.CODE_BASE))
    board.start()
    board.run(50)                       # spins waiting for the input
    assert board.gpio.level(8) is False
    board.gpio.drive_input(7, True)     # the outside world acts
    board.run(200)
    print('\n4. input and edges')
    print(f'   after driving pin 7 high, level(8) = {board.gpio.level(8)}')
    assert board.gpio.level(8) is True


if __name__ == '__main__':
    example_1_pin_state()
    example_2_register_sequence()
    example_3_bit_banged_protocol()
    example_4_input_and_edges()
    print('\nAll examples passed.')
