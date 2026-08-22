# Raspberry Pi emulation

Deep reference for the Raspberry Pi boards in armulator: what is modelled,
how faithfully, where the hardware departs from its own documentation, and
how to validate the models against real silicon.

For the general API see [README.md](README.md). For the Jetson Nano see
[JETSON.md](JETSON.md).

---

## Scope and honest limits

**These boards default to the ARMv6 core** (A32/T32, integer only), while the
Pi 3 (BCM2837, Cortex-A53) and Pi 4 (BCM2711, Cortex-A72) are ARMv8-A. Use
`RaspberryPi3A64` / `RaspberryPi4A64` for the AArch64 core, which does implement
SIMD, the MMU and multi-core — see [AARCH64.md](AARCH64.md).

The Pi-specific documentation here assumes the ARMv6 boards, since that is the
better-travelled path for peripheral work. Either way the peripheral register
interfaces are the same, and that is where GPIO and bus driver logic lives:
writing test firmware as 32-bit ARM (`-marm -march=armv6`) exercises the same
register sequences a production AArch64 driver performs.

**Neither core boots a stock Raspberry Pi OS kernel.** Modelling the CPU is not
modelling the SoC — there is no mailbox, no VideoCore, no SD host, no USB, no
clock manager. A kernel gets nowhere without them.

If you need to boot an OS image, use QEMU's `raspi3b` or `raspi4b` machines
instead. This project is not competing with that and will not get there.

**Throughput is roughly 43,000 instructions/second.** Fine for a few
thousand instructions of driver logic; hopeless for anything resembling a
boot.

---

## Board models

| | Raspberry Pi 3 | Raspberry Pi 4 |
|---|---|---|
| SoC | BCM2837 | BCM2711 |
| Real core | Cortex-A53 ×4 | Cortex-A72 ×4 |
| Peripheral base | `0x3F000000` | `0xFE000000` |
| GPIO pull mechanism | legacy `GPPUD` | `GPIO_PUP_PDN_CNTRL` |
| Interrupt controller | legacy (not modelled) | GIC-400 at `0xFF840000` |

```python
from armulator.boards import RaspberryPi3, RaspberryPi4

board = RaspberryPi4(trace=True)     # trace records every register access
```

### Peripheral map

| Device | Attribute | Pi 3 | Pi 4 |
|---|---|---|---|
| GPIO | `board.gpio` | `0x3F200000` | `0xFE200000` |
| PL011 UART0 | `board.uart` | `0x3F201000` | `0xFE201000` |
| System timer | `board.timer` | `0x3F003000` | `0xFE003000` |
| SPI0 master | `board.spi` | `0x3F204000` | `0xFE204000` |
| BSC1 I²C | `board.i2c` | `0x3F804000` | `0xFE804000` |
| SPI/BSC slave | `board.spi_slave` | `0x3F214000` | `0xFE214000` |
| GIC-400 | `board.gic` | — | `0xFF840000` |

---

## GPIO (BCM2835 / BCM2837 / BCM2711)

54 pins. Register block is identical across all three SoCs **except** the
pull up/down mechanism.

### Register map

| Offset | Register | Notes |
|---|---|---|
| `0x00`–`0x14` | `GPFSEL0`–`GPFSEL5` | 10 pins each, 3 bits per pin |
| `0x1C`, `0x20` | `GPSET0`, `GPSET1` | write-only; read as zero |
| `0x28`, `0x2C` | `GPCLR0`, `GPCLR1` | write-only; read as zero |
| `0x34`, `0x38` | `GPLEV0`, `GPLEV1` | current pin levels |
| `0x40`, `0x44` | `GPEDS0`, `GPEDS1` | event detect status; write 1 to clear |
| `0x4C`, `0x50` | `GPREN0`, `GPREN1` | rising edge enable |
| `0x58`, `0x5C` | `GPFEN0`, `GPFEN1` | falling edge enable |
| `0x64`, `0x68` | `GPHEN0`, `GPHEN1` | high level enable |
| `0x70`, `0x74` | `GPLEN0`, `GPLEN1` | low level enable |
| `0x7C`, `0x80` | `GPAREN0`, `GPAREN1` | async rising edge |
| `0x88`, `0x8C` | `GPAFEN0`, `GPAFEN1` | async falling edge |
| `0x94` | `GPPUD` | **BCM2835/2837 only** |
| `0x98`, `0x9C` | `GPPUDCLK0/1` | **BCM2835/2837 only** |
| `0xE4`–`0xF0` | `GPIO_PUP_PDN_CNTRL_REG0`–`3` | **BCM2711 only** |

### FSEL encoding

Note the non-obvious ordering — ALT4 and ALT5 sit below ALT0:

| Value | Function |
|---|---|
| `0b000` | Input |
| `0b001` | Output |
| `0b010` | ALT5 |
| `0b011` | ALT4 |
| `0b100` | ALT0 |
| `0b101` | ALT1 |
| `0b110` | ALT2 |
| `0b111` | ALT3 |

### The pull up/down trap

This is the most common Pi 3 → Pi 4 porting bug, and the models reproduce
it exactly.

**BCM2835/2837 (legacy):** a two-step clocked handshake. Write the pull
type to `GPPUD`, then write a pin mask to `GPPUDCLK0/1` to commit it. The
`GPPUD` write alone does nothing.

**BCM2711:** direct control. Two bits per pin in
`GPIO_PUP_PDN_CNTRL_REG0`–`3`, no handshake.

**The encodings are also different, and not merely reordered:**

| Bits | Legacy `GPPUD` | BCM2711 `PUP_PDN` |
|---|---|---|
| `0b00` | Off | Off |
| `0b01` | Pull-down | **Pull-up** |
| `0b10` | Pull-up | **Pull-down** |

So a Pi 3 driver run on a Pi 4 fails twice over: the `GPPUD` writes go to a
register BCM2711 does not implement (silently doing nothing), and if you
naively port the constant across you get the opposite pull. Neither failure
raises an error anywhere.

`example/gpio_driver_test.py` scenario 2 demonstrates this.

### Pin-level test API

```python
board.gpio.function(17)        # GpioFunction.OUTPUT
board.gpio.level(17)           # effective electrical level
board.gpio.pull(17)            # Pull.UP / DOWN / OFF
board.gpio.drive_input(7, True)   # external device drives the pin
board.gpio.drive_input(7, None)   # release; falls back to the pull resistor
board.gpio.transitions(17)     # recorded output waveform
board.gpio.pulse_count(17)     # rising edges — useful for PWM/bit-bang tests
```

An output pin reflects what the SoC drives. An input pin reflects the
external driver if one is attached, otherwise its pull resistor. That
distinction matters for interconnect tests, where an unconfigured pin must
float rather than read as low.

---

## GIC-400 (Pi 4 only)

The BCM2711 routes device interrupts through a GIC-400 (GICv2). Interrupts
reach the CPU **only** after firmware enables the distributor and CPU
interface — as on real silicon.

```python
from armulator.peripherals.gic400 import (
    GICC_CTLR, GICC_EOIR, GICC_IAR, GICC_PMR, GICD_CTLR, GICD_ISENABLER,
)

gic = board.gic
gic.write_register(GICD_CTLR, 1)      # enable distributor
gic.write_register(GICC_CTLR, 1)      # enable CPU interface
gic.write_register(GICC_PMR, 0xFF)    # unmask all priorities
gic.write_register(GICD_ISENABLER + 4 * (intid // 32), 1 << (intid % 32))
```

Interrupt IDs follow GICv2 numbering: 0–15 SGI, 16–31 PPI, 32–1019 SPI. A
device at SPI *n* uses ID `32 + n`.

### The acknowledge/EOI subtlety worth knowing

For **level-triggered** interrupts, pending state tracks the input line, and
active/pending are independent states. A source still asserting at EOI
immediately re-presents — which is exactly why a handler must clear the
device *before* writing EOIR. For **edge-triggered**, acknowledging consumes
the latch, so a transient pulse is caught even if it is gone by the time
anyone looks.

Getting this wrong produces either phantom re-fires or lost interrupts.
`tests/test_gic400.py` pins both halves down.

---

## SPI0 master

Standard BCM2835 SPI0. Full duplex: every byte written to `FIFO` is handed
to the selected slave immediately and the returned byte is pushed into the
receive FIFO.

| Offset | Register |
|---|---|
| `0x00` | `CS` — control and status |
| `0x04` | `FIFO` |
| `0x08` | `CLK` — clock divider |
| `0x0C` | `DLEN` |

`CS.TA` (bit 7) frames transfers: asserting and deasserting it drives
chip-select, which slaves that model framing observe.

---

## SPI/BSC slave — the difficult one

`board.spi_slave` models the block that lets a Pi be the *slave* on an SPI
bus. It behaves nothing like a conventional SPI slave, the datasheet chapter
is thin and partly wrong, and it is worth reading this section before
writing any driver against it.

### The dialogue protocol

Transfers are octet-based **dialogues**, MSB first. The first MOSI octet of
each chip-select assertion is an address/direction byte, not data:

- **LSB 0** — write dialogue: subsequent MOSI octets deserialize into the RX
  FIFO; nothing is driven on MISO (it idles high)
- **LSB 1** — read dialogue: subsequent MOSI octets are **discarded**; the TX
  FIFO is serialized out on MISO instead

The upper 7 bits are the slave address, compared against `SLV`. This is I²C
addressing semantics carried onto the SPI side of a shared block.

**Dialogues are half duplex.** You cannot send and receive in one
transaction. A driver expecting a status byte back while sending a command
reads `0xFF` forever.

```python
from armulator.peripherals.spi_slave import (
    CR_EN, CR_RXE, CR_SPI, CR_TXE, address_octet,
)

slave.spi_slave.write_register(0x08, 0x2A)                        # SLV
slave.spi_slave.write_register(0x0C, CR_EN | CR_SPI | CR_RXE | CR_TXE)

master.spi.write_register(0x00, 0x80)                             # TA=1
master.spi.write_register(0x04, address_octet(0x2A, read=False))  # header
master.spi.write_register(0x04, 0x42)                             # payload
master.spi.write_register(0x00, 0x00)                             # TA=0
```

### Register map

| Offset | Register | Notes |
|---|---|---|
| `0x00` | `DR` | **both** FIFO ports: read = RX, write = TX |
| `0x04` | `RSR` | receive status; `OE` = overrun |
| `0x08` | `SLV` | 7-bit slave address |
| `0x0C` | `CR` | control |
| `0x10` | `FR` | flags |
| `0x14` | `IFLS` | FIFO level select |
| `0x18`–`0x24` | `IMSC`, `RIS`, `MIS`, `ICR` | interrupts |
| `0x2C` | `TDR` | test FIFO port |

### Errata modelled

These come from community reverse engineering of real silicon, not the
datasheet. `example/spi_slave_errata.py` reproduces all of them.

1. **`CR.BRK` does not clear the FIFOs.** The datasheet says it does.
   Firmware that sets BRK during init to discard stale TX data will
   transmit that stale data on the next read dialogue. Construct with
   `brk_clears_fifos=True` to model the documented-but-false behaviour, for
   showing a driver depends on a bug.
2. **`TDR` only peeks.** Reading it was suggested as a FIFO-drain
   workaround; it returns the top TX entry without removing it, so the
   workaround does not work.
3. **RX overrun is silent.** Bytes are dropped when the FIFO is full with
   no indication except `RSR.OE`. The dialogue completes normally.
4. **MISO idles high (`0xFF`), not low.**
5. **Startup ordering loses whole transactions.** A master that begins a
   dialogue before the slave's `CR` is written loses everything, with no
   error anywhere.

### Not confirmed

**The interrupt bit assignments in `IMSC`/`RIS`/`MIS`/`ICR` could not be
verified against hardware.** The model uses bit 0 for RX and bit 1 for TX.
If you are testing a driver that depends on the exact layout, verify
`INT_RX` and `INT_TX` in `armulator/peripherals/spi_slave.py` against
silicon first — a passing test here does not settle it.

The FIFO depth is also undocumented; 16 is the community estimate and the
default.

---

## I²C (BSC)

Transactional: set the address in `A`, the byte count in `DLEN`, then set
`ST` in `C`. A read pulls `DLEN` bytes into the FIFO; a write drains the
FIFO to the slave.

Addressing a slave that is not on the bus sets `ERR` in `S` — the NACK path,
which is the branch most drivers get wrong. It is worth asserting on.

---

## Validating against real hardware

Everything above is a model. The replay harness is how you find out whether
the model is right.

### Capturing a trace on a Pi

Linux has MMIO tracepoints on arm64 behind `CONFIG_TRACE_MMIO_ACCESS`. On a
Pi 4 running a 64-bit kernel built with that option:

```bash
cd /sys/kernel/tracing
echo 1 > events/rwmmio/enable
echo > trace
# exercise the driver: toggle a GPIO, run spidev_test, etc.
cat trace > /tmp/capture.txt
```

Lines look like:

```
kworker/0:1-23 [000] ..... 123.456789: rwmmio_write: bcm2835_spi_transfer_one+0x1c/0x2f0 width=32 val=0x80 addr=0xffff800008a04000
```

To narrow the capture to one peripheral, use an ftrace filter on the `addr`
field, or filter after the fact with `Trace.rebase`, which drops anything
outside the window.

If your kernel lacks those tracepoints, capture from userspace via
`/dev/mem` and write the canonical format instead: `op width addr value` per
line.

### Replaying it

```python
from armulator.harness import load, replay_on_board
from armulator.boards import RaspberryPi4

trace = load('capture.txt')
board = RaspberryPi4()
report = replay_on_board(board, 'gpio', trace, captured_base=0xFFFF800008A00000)
print(report.format())
assert report.ok
```

The report gives mismatches with register name, expected vs. actual, and the
kernel function that made the access.

### Reading a report honestly

Three things the report tells you that matter as much as pass/fail:

- **Provenance.** A trace recorded from the model and replayed against the
  model always passes and validates nothing. The report says so explicitly
  when it detects this.
- **Coverage.** A `PASS` on a trace touching 3 of 33 registers is a much
  weaker claim than it looks. Untouched registers are listed.
- **Volatile reads.** Free-running counters and timing-dependent status
  cannot match a capture. Declare them with `volatile={...}`; they are
  executed but not compared, and counted separately so the gap stays
  visible.

### Baseline traces

`traces/` holds traces recorded from the models, regenerated with
`python3 tools/record_baselines.py`. These are **regression guards only** —
they detect behaviour changes, not fidelity problems. Every file is stamped
with that provenance and the harness repeats the warning.

Real hardware captures are the thing that would actually validate this
project. Contributions of them are more valuable than more models.
