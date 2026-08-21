# Description

A pure python ARM emulator

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
SpiBridge(pi, nano, chip_select=0)

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

## Scope

The CPU core is **ARMv6** (A32/T32, integer only). The Pi 3, Pi 4 and Jetson
Nano all ship ARMv8-A cores, so these boards do **not** run AArch64 binaries
or stock vendor kernels — the peripheral *register interfaces* are what is
modelled, since that is where GPIO driver logic lives. Write test firmware as
32-bit ARM (`-marm -march=armv6`) and it exercises the same register
sequences your production driver performs. For booting real OS images, use
QEMU's `raspi3b` / `raspi4b` machines instead.

# Running the tests

Running the tests can be done easily with pytest:

```shell
python3 -m pytest tests -vv
```

# Acknowledgments

* At first, I did it to learn the ARM architecture better. I guess I was carried away.
* Feel free to report bugs.
* Feel free to ask for more features.
