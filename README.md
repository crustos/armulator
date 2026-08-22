# Description

A pure python ARM emulator, with two CPU cores:

- **ARMv6** (`armulator.armv6`) — AArch32, A32 and T32, integer only.
- **AArch64** (`armulator.armv8`) — modelled on the **Cortex-A57**, with all
  four exception levels, SIMD and floating point, two-stage address
  translation, a four-core cluster, a configurable memory model, and AArch32
  execution at EL0. It runs GCC output at every optimisation level, in both
  execution states. See **[AARCH64.md](AARCH64.md)**.

Both sit on the same memory controller and drive the same peripheral models, so
a board can be built around either.

# Installation

Install the last released version using `pip`:

```shell
python3 -m pip install --user -U armulator
```


Or install the latest version from sources:

```shell
git clone git@github.com:matan1008/armulator.git
cd pyiosbackup
python3 -m pip install --user -U -e .
```

# Usage

## ARMv6

To create a processor object, you need to import it first:
```python
from armulator.armv6.arm_v6 import ArmV6
```

Then you can just create it:

```python
arm = ArmV6()
```
Getting familiar with the Memory controller concept is crucial for using the processor.  
In short, there is one "hub" to which you can connect several controllers.  
A "Memory Controller" can be a stick of RAM, Memory mapped LCD screen or whatever you wish.  
  
For example, let's create a RAM controller:

```python
from armulator.armv6.memory_types import RAM
from armulator.armv6.memory_controller_hub import MemoryController

mem = RAM(0x100)
mc = MemoryController(mem, 0xF0000000, 0xF0000100)
arm.mem.memories.append(mc)
```

Now, trying to access a memory between 0xF0000000 and 0xF0000100, will access the `mem` object.  
You can also change the memory manually:

```python
mem.write(0, 2, "\xfe\xe7")
```

Another useful feature is playing with the memory protection or management unit,
for example cancelling memory protection will look like:
```python
arm.registers.sctlr.m = 0
arm.take_reset()
```
Please note that after changing internal features it is recommended to reset the processor.  
  
When running the armulator, you will probably want to start from a defined address, so:
```python

arm.registers.branch_to(0x100)
```

The last thing we need to do is to really run the processor, which can be done with:
```python
arm.emulate_cycle()
```

## AArch64

The AArch64 core takes its memory map at construction and needs no global
configuration:

```python
from armulator.armv8.arm_v8 import ArmV8

cpu = ArmV8([{'mem_type': 'RAM', 'beginning': 0x0, 'end': 0x10000}])
cpu.take_reset()
cpu.registers.branch_to(0x1000)
cpu.emulate_cycle()
```

Registers differ from AArch32 in ways worth knowing before you start. `X0`–`X30`
are flat, with no banking by mode. Register 31 is context dependent — zero in
most instructions, `SP` in a few — so you say which you mean:

```python
cpu.registers.get_x(31)          # always 0, this is XZR
cpu.registers.get_reg_or_sp(31)  # the stack pointer
```

A 32-bit write zeroes the upper half of its destination, and the same rule
applies to the vector registers: writing `D0` clears bits 127:64 of `V0`.

Reset leaves the core at EL1 with the MMU off and SIMD trapped. Firmware enables
what it needs, as it would on hardware — see
[AARCH64.md](AARCH64.md) for translation, exception levels, multi-core and the
memory model.


# Board and peripheral emulation

For testing GPIO and peripheral driver logic, `armulator.boards` provides
prebuilt machines with peripherals mapped at the real SoC addresses:

```python
from armulator.boards import RaspberryPi4
from armulator.boards.firmware import firmware

board = RaspberryPi4(trace=True)
board.load(board.CODE_BASE, firmware("""
    ldr r0, =0xFE200000
    mov r1, #1
    lsl r1, r1, #21        @ FSEL17 = output
    str r1, [r0, #0x04]    @ GPFSEL1
    mov r2, #1
    lsl r2, r2, #17
    str r2, [r0, #0x1C]    @ GPSET0
""", address=board.CODE_BASE))
board.start()
board.run()

assert board.gpio.level(17) is True
print(board.format_trace())     # gpio.GPFSEL1 <- 0x00200000 ...
```

Available boards: `RaspberryPi3` (BCM2837, peripherals at `0x3F000000`),
`RaspberryPi4` (BCM2711, `0xFE000000`) and `JetsonNano` (Tegra X1, GPIO at
`0x6000D000`).

Peripherals expose a pin-level API so tests can act as the outside world:

```python
board.gpio.drive_input(7, True)   # external device pulls pin 7 high
board.gpio.level(7)               # effective level
board.gpio.function(7)            # GpioFunction.INPUT / OUTPUT / ALT0 ...
board.gpio.pull(7)                # Pull.UP / DOWN / OFF
board.gpio.transitions(17)        # recorded output waveform
board.pending_irq()               # devices asserting their IRQ line
```

Writing peripherals of your own means subclassing `MMIODevice` and
implementing two methods:

```python
from armulator.peripherals import MMIODevice

class MyDevice(MMIODevice):
    REGISTERS = {0x00: 'CTRL', 0x04: 'STATUS'}

    def read_register(self, offset): ...
    def write_register(self, offset, value): ...

board.attach('mydev', MyDevice(0x1000), offset=0x100000)
```

## Interrupt routing (GIC-400)

The Pi 4 and Jetson Nano boards include a GIC-400 interrupt controller, with
device lines wired to their SoC SPI numbers. Interrupts only reach the CPU
once firmware enables the distributor and CPU interface, as on real silicon:

```python
from armulator.peripherals.gic400 import (
    GICC_CTLR, GICC_PMR, GICD_CTLR, GICD_ISENABLER, GICC_IAR, GICC_EOIR,
)

gic = board.gic
gic.write_register(GICD_CTLR, 1)          # enable distributor
gic.write_register(GICC_CTLR, 1)          # enable CPU interface
gic.write_register(GICC_PMR, 0xFF)        # unmask all priorities
gic.write_register(GICD_ISENABLER + 4, 1 << 17)

intid = gic.read_register(GICC_IAR)       # acknowledge
...                                        # service the device
gic.write_register(GICC_EOIR, intid)      # end of interrupt
```

Priority, edge/level configuration and the acknowledge/EOI handshake are all
modelled, including the case that catches real drivers out: a level-triggered
source that is still asserting at EOI immediately re-presents.

The Pi 3 has no GIC (the BCM2837 uses the legacy controller), so it falls
back to polling device lines directly.

## Motor control

`armulator.peripherals` models a motor HAT end to end — a PCA9685 PWM controller
on I2C, driving H-bridges, driving DC motors or steppers:

```python
from armulator.boards import RaspberryPi4
from armulator.peripherals import MotorHat

board = RaspberryPi4()
hat = MotorHat().attach_to(board)      # PCA9685 on the Pi's I2C bus at 0x60
motor = hat.attach_dc_motor(1)

# ... firmware runs and programs the controller ...

hat.advance(1.0)                       # one second of shaft time
assert motor.position > 0
```

Because the layers are separate, a test asserts on where the shaft ended up
rather than on the register writes that were meant to move it. The full
walkthrough, including the three PCA9685 behaviours that silently defeat a
plausible-looking driver, is in [MOTOR.md](MOTOR.md).

## Serial buses

`Bcm2835Spi` and `Bcm2835I2c` are available on all boards as `board.spi` and
`board.i2c`. Slaves are plain Python objects:

```python
from armulator.peripherals.serial_bus import I2cSlaveDevice, SpiSlaveDevice

sensor = board.i2c.attach_slave(I2cSlaveDevice(address=0x48, registers={0: 0xDE}))
board.spi.attach_slave(SpiSlaveDevice(responses=b'\x99'), chip_select=0)
```

I2C models the NACK path — addressing a slave that isn't on the bus sets ERR
in the status register.

## SPI slave mode

`Bcm2835SpiSlave` models the Pi's SPI/BSC slave block (`board.spi_slave`),
and it is deliberately faithful to a peripheral that behaves nothing like a
conventional SPI slave:

* transfers are **half duplex** octet "dialogues", not simultaneous
  shift-in/shift-out
* each dialogue opens with an **address/direction octet** — upper 7 bits are
  the slave address, LSB 0 selects write and LSB 1 selects read
* during a write dialogue MISO idles high and the TX FIFO is untouched

```python
from armulator.peripherals.spi_slave import CR_EN, CR_RXE, CR_SPI, CR_TXE, address_octet

slave.spi_slave.write_register(0x08, 0x2A)                        # SLV
slave.spi_slave.write_register(0x0C, CR_EN | CR_SPI | CR_RXE | CR_TXE)

master.spi.write_register(0x00, 0x80)                             # CS: TA=1
master.spi.write_register(0x04, address_octet(0x2A, read=False))  # header
master.spi.write_register(0x04, 0x42)                             # payload
master.spi.write_register(0x00, 0x00)                             # CS: TA=0

assert slave.spi_slave.received == b'\x42'
```

Several hardware errata are modelled, since they are what catch driver
authors out: `CR.BRK` does not actually clear the FIFOs (set
`brk_clears_fifos=True` to get the behaviour the datasheet *describes*),
`TDR` only peeks at the TX FIFO rather than draining it, and RX overrun is
silent apart from `RSR.OE`. The datasheet's interrupt bit assignments for
this block could not be confirmed against hardware — see the module
docstring before relying on them.

`example/spi_slave_errata.py` walks through five mistakes that look correct
in review and fail on silicon.

## Emulating several boards together

`armulator.boards.interconnect` wires boards to each other, so a Pi and a
Jetson can exchange signals the way they would on a bench:

```python
from armulator.boards import JetsonNano, RaspberryPi4
from armulator.boards.interconnect import GpioLink, Machine, SpiBridge

machine = Machine()
pi = machine.add('pi', RaspberryPi4())
nano = machine.add('nano', JetsonNano())

machine.link(GpioLink(pi, 17, nano, 'PA0', name='DATA'))    # Pi drives
machine.link(GpioLink(nano, 'PA1', pi, 27, name='READY'))   # Jetson answers
SpiBridge(pi, pi3, chip_select=0)    # slave end needs a spi_slave block

machine.run_until(lambda: pi.cpu.registers.get(4) == 1)
```

Boards are stepped round-robin in instruction slices with all links settled
between slices — deterministic and repeatable, but not cycle-accurate, since
these are independent cores with independent clocks. Only pins configured as
outputs drive a wire; an input pin releases it, and the receiver falls back
to its own pull resistor.

See `example/two_device_link.py` for a full walkthrough including a two-way
handshake that neither board can complete alone.

Assembling test firmware from source requires `keystone-engine`
(`pip install keystone-engine`); pre-assembled bytes work without it.

See `example/gpio_driver_test.py` for a worked walkthrough.

## Choosing a core

Every board takes an `arch=` argument, and the `*A64` classes are the same
boards built around the AArch64 core:

| Board | Core | Notes |
|---|---|---|
| `RaspberryPi3`, `RaspberryPi4`, `JetsonNano` | ARMv6 | A32/T32 firmware |
| `RaspberryPi3A64`, `RaspberryPi4A64`, `JetsonNanoA64` | AArch64 | single core |
| `JetsonNanoA64Smp` | AArch64 | the full quad-core A57 cluster |

The ARMv6 boards remain the default. They are the older and better-travelled
path, and the peripheral *register interfaces* — where GPIO driver logic
actually lives — are identical either way, so A32 test firmware exercises the
same sequences a production AArch64 driver performs.

Reach for the AArch64 boards when the code under test is AArch64: compiler
output, code that uses the MMU or several cores, or anything where memory
ordering matters.

```python
from armulator.boards import JetsonNanoA64
from armulator.boards.firmware import firmware_a64

board = JetsonNanoA64()
board.load(board.CODE_BASE, firmware_a64('''
        movz x0, #0x6000, lsl #16
        movk x0, #0xD000
        movz w1, #1
        str  w1, [x0, #0x00]        // CNF port A -> GPIO
        str  w1, [x0, #0x10]        // OE  -> output
        str  w1, [x0, #0x20]        // OUT -> high
''', address=board.CODE_BASE))
board.start()
board.run(200)
assert board.gpio.level('PA0') is True
```

Neither core boots stock vendor kernels — the peripheral coverage is nowhere
near a whole SoC. For booting real OS images, use QEMU's `raspi3b` / `raspi4b`
machines instead.

# Validating models against hardware

`armulator.harness` replays a captured driver trace against a peripheral
model and asserts every read returns what the silicon returned. It is the
strongest correctness check available without owning the board:

```python
from armulator.boards import RaspberryPi4
from armulator.harness import load, replay_on_board

trace = load('capture.txt')          # ftrace rwmmio or canonical format
report = replay_on_board(RaspberryPi4(), 'gpio', trace,
                         captured_base=0xFFFF800008A00000)
print(report.format())
assert report.ok
```

Reports carry three things beyond pass/fail: **provenance** (a trace
recorded from the model and replayed against it is circular — the report
says so), **coverage** (a PASS on 3 of 33 registers is a weak claim, so
untouched registers are listed), and **volatile reads** (counters that
cannot match a capture are executed but not compared, and counted).

Capture recipe and the full workflow are in [RASPI.md](RASPI.md#validating-against-real-hardware).
`example/replay_driver_trace.py` walks through it.

`traces/` holds baselines recorded from the models — regression guards only,
not hardware validation. Regenerate with `python3 tools/record_baselines.py`.

# Further reading

- **[RASPI.md](RASPI.md)** — Raspberry Pi register maps, the Pi 3 → Pi 4
  pull up/down trap, GIC-400 acknowledge/EOI semantics, the SPI slave
  dialogue protocol and its errata, and the hardware capture recipe.
- **[JETSON.md](JETSON.md)** — Tegra GPIO structure and masked registers,
  the Tegra SPI controller's triggered-transfer model and its two
  off-by-one register traps, plus an explicit list of what is missing.
- **[MOTOR.md](MOTOR.md)** — driving motors: a PCA9685 on I2C feeding H-bridges
  feeding DC motors and steppers, so a test can assert on shaft position rather
  than on register writes.
- **[AARCH64.md](AARCH64.md)** — the AArch64 core: instruction coverage,
  exception levels and routing, two-stage translation, the multi-core cluster
  and PSCI bring-up, and the memory model that makes a missing barrier
  observable.

# Running the tests

Running the tests can be done easily with pytest:

```shell
python3 -m pytest tests -vv
```

# Acknowledgments

* At first, I did it to learn the ARM architecture better. I guess I was carried away.
* Feel free to report bugs.
* Feel free to ask for more features.
