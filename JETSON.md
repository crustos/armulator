# Jetson Nano emulation

Reference for the NVIDIA Jetson Nano (Tegra X1 / T210) board model: what is
modelled, what is a stand-in, and what is missing outright.

For the general API see [README.md](README.md). For the Raspberry Pi see
[RASPI.md](RASPI.md).

**Read the gaps section before relying on this.** Jetson support is
meaningfully thinner than the Pi support, and the difference is not obvious
from the API — both boards look the same from the outside.

---

## Scope and honest limits

**The CPU can be either core.** The Jetson Nano's real processor is a quad
Cortex-A57 (ARMv8-A), and that is now modelled: `JetsonNanoA64` gives one A57,
`JetsonNanoA64Smp` the full four-core cluster. `JetsonNano` remains the ARMv6
board, and remains the default. See [choosing a core](#choosing-a-core) below.

**The vendor kernel still will not boot.** Modelling the CPU is not the same as
modelling the SoC. Four peripherals out of the T210's several dozen are present
— no memory controller, no clock and reset controller, no power management, no
display, no USB. A kernel gets nowhere without them.

**The boot chain is out of scope and will stay that way.** The Tegra X1
boots through a signed chain — boot ROM, then CBoot — that is proprietary
and largely undocumented. There is no path to emulating it here, and no
emulator does. If you need to run Jetson software, use real hardware.

**The GPU does not exist here.** The Nano's 128-core Maxwell GPU is the
reason most people buy the board. None of it is modelled, and CUDA
workloads are entirely out of reach. If your interest in the Nano is the
GPU, this project has nothing for you.

---

## What is actually modelled

| Device | Attribute | Address | Fidelity |
|---|---|---|---|
| Tegra GPIO | `board.gpio` | `0x6000D000` | **Real model** |
| UART A | `board.uart` | `0x70006000` | PL011 model, close enough |
| SPI1 | `board.spi` | `0x7000D400` | **Real model** |
| GIC-500 | `board.gic` | `0x50041000` | GICv2-compatible |

```python
from armulator.boards import JetsonNano

board = JetsonNano(trace=True)
```

Memory layout differs from the Pi boards: RAM is based at `0x80000000` and
code loads at `0x80080000`, matching where a Tegra kernel image lands.

---

## Choosing a core

| Class | Cores | Firmware |
|---|---|---|
| `JetsonNano` | 1 × ARMv6 | A32/T32 |
| `JetsonNanoA64` | 1 × Cortex-A57 | A64 |
| `JetsonNanoA64Smp` | 4 × Cortex-A57 | A64 |

The peripheral models are identical across all three — same addresses, same
registers, same semantics. Only the processor differs, so A32 test firmware
exercises exactly the register sequences a production AArch64 driver performs.

The same GPIO sequence, written for each core:

```python
# ARMv6
from armulator.boards import JetsonNano
from armulator.boards.firmware import firmware

board = JetsonNano()
board.load(board.CODE_BASE, firmware('''
        ldr r0, =0x6000D000
        mov r1, #1
        str r1, [r0, #0x00]        @ CNF port A -> GPIO
        str r1, [r0, #0x10]        @ OE  -> output
        str r1, [r0, #0x20]        @ OUT -> high
''', address=board.CODE_BASE))
```

```python
# AArch64
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
```

Both then `board.start()`, `board.run(200)`, and leave `PA0` high.

### Peripherals through the MMU

With translation enabled the GPIO block can sit at a kernel virtual address, and
the peripheral window should be mapped as **Device** memory in `MAIR` so
accesses are neither buffered nor reordered:

```python
board = JetsonNanoA64(ram_size=0x400000)
# ... build tables: RAM identity-mapped as Normal, 0x6000D000 mapped at
#     0xC0000000 as Device ...
board.load(board.CODE_BASE, firmware_a64('''
        mrs  x3, sctlr_el1
        orr  x3, x3, #1              // SCTLR_EL1.M
        msr  sctlr_el1, x3
        isb
        movz x0, #0xC000, lsl #16    // the GPIO block, virtually
        movz w1, #1
        str  w1, [x0, #0x00]
''', address=board.CODE_BASE))
```

Code must be identity-mapped across the instruction that enables the MMU, or the
next fetch lands somewhere unmapped. `tests/test_a64_mmu_firmware.py` is a
working example.

### Bringing up the other three cores

Secondary cores start parked, as they do on hardware, and are released through
PSCI — the same `SMC` calls the vendor firmware answers:

```python
board = JetsonNanoA64Smp()
board.cluster.slice_size = 5
```

```asm
        movz x20, #1                 // start with core 1
next:   movz x0, #0x0003
        movk x0, #0xC400, lsl #16    // PSCI CPU_ON
        mov  x1, x20                 // target MPIDR (Aff0 = core number)
        movz x2, #...                // entry point
        mov  x3, x20                 // context id, arrives in the core's x0
        smc  #0
        add  x20, x20, #1
        cmp  x20, #4
        b.lo next
```

Each core reads its own number from `MPIDR_EL1`:

```asm
        mrs  x10, mpidr_el1
        and  x10, x10, #0xFF         // Aff0
```

All four share the GPIO controller, so concurrent access needs a lock —
`LDAXR`/`STLXR` to acquire and `STLR` to release. A plain `STR` release looks
correct under the default memory model and is not; see
[AARCH64.md](AARCH64.md#the-memory-model) for how to make that failure visible.
`tests/test_a64_smp_firmware.py` runs four cores toggling `PA0` under a lock.

---

## Tegra GPIO — the real model

This is the part of the Jetson support that is genuinely faithful, and it is
organised very differently from Broadcom's.

### Structure

8 controllers × 4 ports × 8 pins = **256 pins**. Addressing:

```
controller_base = 0x6000D000 + controller * 0x100
register_addr   = controller_base + reg_offset + port * 4
```

Each register is replicated once per port at 4-byte intervals within a
controller, and controllers are `0x100` apart.

### Register map (per controller)

| Offset | Register | Purpose |
|---|---|---|
| `0x00` | `CNF` | 1 = GPIO mode, 0 = SFIO (pin mux) |
| `0x10` | `OE` | output enable |
| `0x20` | `OUT` | output value |
| `0x30` | `IN` | input value |
| `0x40` | `INT_STA` | interrupt status |
| `0x50` | `INT_ENB` | interrupt enable |
| `0x60` | `INT_LVL` | interrupt level/edge select |
| `0x70` | `INT_CLR` | interrupt clear |
| `0x80` | `MSK_CNF` | masked alias of `CNF` |
| `0x90` | `MSK_OE` | masked alias of `OE` |
| `0xA0` | `MSK_OUT` | masked alias of `OUT` |
| `0xC0`–`0xE0` | `MSK_INT_*` | masked aliases |

### The three-gate rule

Unlike Broadcom, where FSEL alone selects output, a Tegra pin needs **both**
`CNF` (GPIO mode, not SFIO) and `OE` (output enable) before `OUT` drives
anything. Writing `OUT` to a pin still in SFIO mode does nothing, silently.

```python
gpio.write_register(0x00, 1 << 0)   # CNF: PA0 to GPIO mode
gpio.write_register(0x10, 1 << 0)   # OE:  output
gpio.write_register(0x20, 1 << 0)   # OUT: high
```

### Masked registers

The `MSK_*` aliases at `+0x80` are the important quirk. Writing them uses
the **upper 8 bits as a write-enable mask** for the lower 8 bits, letting
firmware update one pin without a read-modify-write:

```python
# Set PA3 high, leave every other pin in port A untouched.
gpio.write_register(0xA0, (1 << (8 + 3)) | (1 << 3))
```

Linux's `gpio-tegra` driver uses these almost exclusively, so any realistic
driver test needs them. Reads of a `MSK_*` register behave like its plain
alias.

### Pin naming

```python
TegraGpio.pin_number('PA0')    # 0
TegraGpio.pin_number('PB0')    # 8
TegraGpio.pin_number('PZ7')    # 207
TegraGpio.pin_number('PAA0')   # 208  (doubled letters continue past Z)
TegraGpio.pin_number('PBB4')   # 220
```

Most pin-level API calls accept either a flat index or a name string:

```python
board.gpio.level('PA0')
board.gpio.drive_input('PBB4', True)
board.gpio.is_gpio('PA0')      # False if the pin is still SFIO
board.gpio.is_output('PA0')    # needs both CNF and OE
```

---

## SPI1 — Tegra X1 controller

`board.spi` models the Tegra SPI block at `0x7000D400`, the controller the
Jetson's 40-pin header exposes. The register map and bit definitions follow
the in-tree drivers (U-Boot `drivers/spi/tegra114_spi.c` and Linux
`drivers/spi/spi-tegra114.c`), which cover T114/T124/T210/T186 with the same
block. Those are a better source than the datasheet here, since the Tegra X1
TRM is available only to registered developers.

### Register map

| Offset | Register | Purpose |
|---|---|---|
| `0x000` | `COMMAND1` | word length, master/slave, TX/RX enable, CS, start |
| `0x004` | `COMMAND2` | tap delays |
| `0x008`, `0x00C` | `CS_TIM1`, `CS_TIM2` | chip select timing |
| `0x010` | `TRANS_STATUS` | block count, `RDY` |
| `0x014` | `FIFO_STATUS` | FIFO flags, counts, error bits, flush |
| `0x018`, `0x01C` | `TX_DATA`, `RX_DATA` | single-word ports |
| `0x020`, `0x024` | `DMA_CTL`, `DMA_BLK` | block count minus one |
| `0x108` | `TX_FIFO` | transmit FIFO port |
| `0x188` | `RX_FIFO` | receive FIFO port |

### Transfers are triggered, not implied

This is the biggest behavioural difference from Broadcom, and the thing that
breaks ported firmware. On the BCM2835 controller a FIFO write shifts a byte
immediately. On Tegra, nothing touches the bus until `COMMAND1.PIO` (bit 31)
is set:

```python
from armulator.peripherals.spi_tegra import (
    CMD1_M_S, CMD1_PIO, CMD1_RX_EN, CMD1_TX_EN,
    SPI_COMMAND1, SPI_DMA_BLK, SPI_RX_FIFO, SPI_TX_FIFO,
)

cmd = CMD1_M_S | CMD1_TX_EN | CMD1_RX_EN | 7    # master, 8-bit words
spi.write_register(SPI_COMMAND1, cmd)
spi.write_register(SPI_TX_FIFO, 0x42)
spi.write_register(SPI_DMA_BLK, 0)              # one packet (count - 1)
spi.write_register(SPI_COMMAND1, cmd | CMD1_PIO)   # go
```

A driver ported from a Pi fills the TX FIFO and then waits forever. The
model reproduces that rather than papering over it —
`test_broadcom_firmware_does_not_work_on_tegra` asserts it explicitly.

### Two more off-by-one traps

Both registers store **one less** than the value you want:

- `COMMAND1.BIT_LENGTH` holds bits minus one — an 8-bit word is `7`
- `DMA_BLK` holds packet count minus one — one packet is `0`

### Chip select framing

A `PIO` trigger asserts chip select for the duration of the transfer and
releases it afterwards (unless `CS_SW_HW` holds it), so slaves that frame
their protocol on CS see one dialogue per trigger. That is what lets a Tegra
master drive a Pi's SPI slave block correctly:

```python
nano, pi = JetsonNano(), RaspberryPi3()
SpiBridge(nano, pi)         # Tegra master -> BCM2835 slave
```

### Modelled but simplified

- Only whole-byte word lengths move data; `bit_length` records what was
  programmed, but sub-byte and >8-bit words are truncated to bytes
- Slave mode (`COMMAND1.M_S` clear) is not modelled — a transfer simply does
  not happen, rather than the controller pretending to be a master
- DMA (`DMA_CTL`) is stored but does nothing; transfers are PIO only
- Packed mode (`COMMAND1.PACKED`) is accepted and ignored

## Gaps you need to know about

### There is no Tegra SPI slave

The Jetson has no modelled SPI slave controller at all. `SpiBridge` raises
`ValueError` if you try to use a Jetson as the slave end, rather than
silently faking it:

```python
SpiBridge(pi, nano)      # ValueError: JetsonNano has no spi_slave controller
```

For cross-device SPI, use a Pi as the slave end — the BCM2835 SPI slave is a
real model. See [RASPI.md](RASPI.md).

### Not modelled at all

- I²C (Tegra I2C controller)
- PWM
- DMA (APB DMA)
- Pinmux controller (`0x70000000`) — `CNF` selects GPIO vs SFIO, but which
  SFIO function a pin takes is not modelled
- Clock and Reset Controller (CAR)
- Power management (PMC)
- Memory controller
- Everything GPU-related

### GIC-500 and multi-core interrupts

Modelled with the `Gic400` class. GIC-500 is GICv2-compatible at the
register level for the distributor and CPU interface, which is the part
that matters here. GICv3 features (affinity routing, ITS, system register
access) are not modelled. The SPI numbers wired up
(`GPIO_SPI`, `UART_SPI`, `SPI_SPI`) are placeholders rather than verified
Tegra X1 interrupt assignments.

On `JetsonNanoA64Smp` the CPU interface registers (`GICC_CTLR`, `GICC_PMR`) are
banked per core, so each core enables its own interface. Peripheral interrupts
are routed by the distributor's target byte, and SGIs work as inter-processor
interrupts including broadcast to several cores at once:

```python
gic.write_register(GICD_SGIR, (target_mask << 16) | sgi_id)
```

An interrupt wakes a core out of `WFI` whether or not it is unmasked, since the
wake-up condition is the interrupt arriving rather than it being taken.

Interrupt routing across exception levels follows `HCR_EL2.IMO` and
`SCR_EL3.IRQ` — a hypervisor or secure firmware can claim a guest's interrupts.
See [AARCH64.md](AARCH64.md#registers-and-exception-levels).

---

## Cross-device work

Where the Jetson support is genuinely useful today is GPIO-level
interconnect with a Pi:

```python
from armulator.boards import JetsonNano, RaspberryPi4
from armulator.boards.interconnect import GpioLink, Machine

machine = Machine()
pi = machine.add('pi', RaspberryPi4())
nano = machine.add('nano', JetsonNano())

machine.link(GpioLink(pi, 17, nano, 'PA0', name='DATA'))
machine.link(GpioLink(nano, 'PA1', pi, 27, name='READY'))

machine.run_until(lambda: pi.cpu.registers.get(4) == 1)
```

`GpioLink` handles the Broadcom/Tegra difference automatically — it checks
`is_output` on the Tegra side and `function` on the Broadcom side. Only pins
configured as outputs drive a wire; an input pin releases it, and the
receiver falls back to its own pull resistor.

`example/two_device_link.py` walks through a two-way handshake that neither
board can complete alone, which is the test that fails loudly if the
interconnect ever stops propagating.

---

## Validating against real hardware

The replay harness works the same way as for the Pi; see the
[validation section in RASPI.md](RASPI.md#validating-against-real-hardware)
for the capture recipe.

Two Jetson-specific notes:

- Tegra register offsets in a capture need rebasing onto `0x6000D000`, and
  the controller stride means a capture spanning several controllers should
  use a `span` covering all of them rather than a single `0x100`.
- `board.spi` replays are now meaningful — the register map follows the
  in-tree `tegra114` driver. A capture from `spi-tegra114` on a Nano
  would validate it.

A real `gpio-tegra` capture from a Nano would be the single most valuable
contribution to this file — it would validate the one part of the Jetson
support that claims fidelity.
