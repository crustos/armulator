# AArch64 emulation

`armulator.armv8` is a second CPU core living beside the ARMv6 one. It models
AArch64 as implemented by the **Cortex-A57** — the core in the Tegra X1, and so
in the Jetson Nano.

For the general API see [README.md](README.md); for the boards built around this
core see [JETSON.md](JETSON.md) and [RASPI.md](RASPI.md).

It is a separate package rather than an extension of `armulator.armv6`, because
AArch64 is not a superset of AArch32. There is no register banking by processor
mode, no condition field on ordinary instructions, no Thumb, and the seven
processor modes are replaced by four exception levels. What the two share —
the memory controller hub, the peripheral models, the bit primitives — is
reused directly.

```python
from armulator.armv8.arm_v8 import ArmV8

cpu = ArmV8([{'mem_type': 'RAM', 'beginning': 0x0, 'end': 0x10000}])
cpu.take_reset()
cpu.registers.branch_to(0x1000)
cpu.emulate_cycle()
```

## Contents

- [What runs](#what-runs)
- [Instruction coverage](#instruction-coverage)
- [Registers and exception levels](#registers-and-exception-levels)
- [Floating point and SIMD](#floating-point-and-simd)
- [Address translation](#address-translation)
- [AArch32 at EL0](#aarch32-at-el0)
- [Multi-core](#multi-core)
- [The memory model](#the-memory-model)
- [What is not modelled](#what-is-not-modelled)

## What runs

Compiler output. The core has been checked against GCC-generated AArch64 at
`-O0`, `-O1`, `-O2`, `-O3` and `-Os`: every instruction emitted decodes, and
compiled functions produce the same answers as the C they came from — CRC-32
matching zlib, a 3×3 matrix multiply, FNV-1a hashing, floating point loops, and
struct-by-value calls that use the homogeneous-aggregate calling convention.

That is the bar worth stating, because a decoder can report high coverage while
quietly executing something wrong. Decoding an instruction and executing it
correctly are different claims.

## Instruction coverage

Every one of the ten top-level A64 encoding groups decodes. The top level
dispatches on a single 4-bit field at `instr[28:25]`, which makes the tree far
more regular than the AArch32 one.

| Group | Coverage |
|---|---|
| Data processing — immediate | ADR/ADRP, add/subtract, logical, move wide, bitfield, extract |
| Data processing — register | Shifted and extended arithmetic, logical, ADC/SBC, conditional select and compare, divides, variable shifts, RBIT/REV/CLZ, multiplies |
| Branches, exception, system | B/BL/B.cond, CBZ/TBZ, BR/BLR/RET/ERET, SVC/HVC/SMC/BRK, hints, barriers, MRS/MSR, cache and TLB maintenance |
| Loads and stores | All widths and addressing modes, pairs, literals, exclusives, acquire/release |
| SIMD and floating point | See [below](#floating-point-and-simd) |

Unimplemented encodings decode as **undefined** rather than executing as
something else. This matters more than it sounds: an instruction that decodes
to the wrong class runs silently and produces wrong answers, which is worse
than one that faults. See the note on `USHR` in
[what is not modelled](#what-is-not-modelled).

## Registers and exception levels

`X0`–`X30` are flat — no banking by mode. Register 31 is context dependent: it
reads as zero in most instructions and denotes `SP` in a handful, so callers say
which they mean rather than the register file guessing:

```python
cpu.registers.get_x(31)          # always 0 — XZR
cpu.registers.get_reg_or_sp(31)  # the stack pointer
```

A 32-bit write **zeroes the upper half** of the destination. This is the single
most common source of AArch64 emulation bugs and is covered by tests at both the
register-file and instruction level.

All four exception levels are modelled, with per-level stack pointers and
exception-return state (`ELR`, `SPSR`, `ESR`, `FAR`, `VBAR`). PSTATE is held as
discrete fields and packed into an SPSR only when an exception is taken.

**Exception routing** is where the level hierarchy earns its keep. Which level
takes an exception depends on configuration, not just on the instruction:

| Condition | Target |
|---|---|
| `SVC` | EL1, or EL2 when `HCR_EL2.TGE` is set |
| `HVC` | EL2 |
| `SMC` | EL3 |
| IRQ with `HCR_EL2.IMO` | EL2 — the hypervisor claims the guest's interrupts |
| IRQ with `SCR_EL3.IRQ` | EL3 — secure firmware claims them instead |
| Stage 2 fault | EL2, always |

Two rules apply throughout: an exception never targets a level below the one it
came from, and never one that is not implemented. A core built with
`ArmV8(highest_el=EL.EL2)` routes `SMC` to EL2 instead.

`ERET` is also the boot path: write the level you want into `SPSR_EL3`, point
`ELR_EL3` at the entry point, and execute it.

Security state follows `SCR_EL3.NS`, with EL3 always secure. Reset is to secure
state, as on hardware.

## Floating point and SIMD

32 × 128-bit `V` registers, `FPCR`/`FPSR`, scalar single and double precision
arithmetic, compares, `FCSEL`, conversions, and the SIMD/FP load-store forms
including 128-bit `Q` accesses and `LD1`/`ST1`. Vector arithmetic covers
three-same integer and floating point operations, saturating add and subtract,
bitwise operations, immediates, `DUP`/`INS`/`UMOV`/`SMOV`, shifts by immediate,
by-element multiply, pairwise `ADDP`, and the across-lane reductions
`ADDV`/`SMAXV`/`UMAXV`/`SMINV`/`UMINV`.

`LD1`–`LD4` and `ST1`–`ST4` move arrays of structures, de-interleaving on the
way: `LD2` reading eight words puts the even ones in the first register and the
odd ones in the second, turning an interleaved array of pairs into two parallel
vectors. Saturating arithmetic clamps rather than wrapping, which is the point
of it — an overflowing sample pins at full scale instead of flipping sign.

Writing a narrow view of a `V` register **zeroes the rest**, mirroring the
32-bit rule for `X`: writing `D0` clears bits 127:64 of `V0`.

IEEE-754 behaviour is modelled properly rather than deferred to Python's
defaults — signed infinities, NaN propagation with signalling NaNs quietened,
saturation rather than wrapping on out-of-range conversions, NaN converting to
zero, and sign-preserving `sqrt(-0.0)`.

**One deliberate piece of fidelity:** `CPACR_EL1` resets to trapping SIMD and
floating point, so firmware must enable it before touching a vector register:

```asm
    mov  x0, #0x300000        // CPACR_EL1.FPEN
    msr  cpacr_el1, x0
```

Skipping this faults with `ESR.EC = 0x07`, exactly as on hardware, rather than
silently working.

## Address translation

Stage 1 translation for all three regimes (EL1&0, EL2, EL3), with 4KB, 16KB and
64KB granules, 1GB and 2MB blocks and 4KB pages, `TTBR0`/`TTBR1` selection,
AP permissions, `UXN`/`PXN`/`WXN`, the access flag, `APTable` restrictions that
accumulate down the walk, top-byte-ignore, and `MAIR` attribute decoding.

Faults carry the architectural status in `ESR`, so a handler can tell a missing
translation (`0b0001LL`) from a permission fault (`0b0011LL`) from a clear
access flag (`0b0010LL`), with the level in the low bits.

**Stage 2** turns what a guest believes is physical memory into an intermediate
address the hypervisor resolves. The guest cannot see or change these tables,
which is what makes the isolation hold:

```
guest VA 0x40000000  ──stage 1──▶  IPA 0x200000  ──stage 2──▶  PA 0x900000
        (guest tables)                    (VTTBR_EL2, invisible to the guest)
```

Stage 2 permissions override the guest's own: a page the guest's tables call
writable is read-only if stage 2 says so. A stage 2 fault is reported to EL2
with `FAR_EL2` holding the guest's *virtual* address and `HPFAR_EL2` the
intermediate one.

The TLB exists for speed, not fidelity — a walk is several Python memory reads,
and doing that per access makes even small firmware crawl. It is keyed by regime
and security state, and flushed on `TLBI` and on writes to translation control
registers. **Stale entries persist until invalidated**, as on real hardware, and
there is a test asserting exactly that.

## AArch32 at EL0

The Cortex-A57 can execute 32-bit code at EL0, which is how a 64-bit kernel runs
32-bit applications:

```python
cpu.enter_aarch32(entry_point)          # or thumb=True for T32
cpu.emulate_cycle()
```

The instruction set involved is the same A32/T32 that `armulator.armv6` already
implements in full, so this is an **adapter rather than a second
implementation**: the ARMv6 opcodes execute unchanged against a view that
presents their expected interface, backed by AArch64 state. Reimplementing them
here would have duplicated several hundred opcodes and then drifted out of step.

The register mapping is not a convenience — it is what the architecture
specifies:

| AArch32 | AArch64 |
|---|---|
| `R0`–`R12` | `X0`–`X12`, low 32 bits |
| `SP` (`R13`) | `X13` |
| `LR` (`R14`) | `X14` |
| `CPSR.NZCV` | `PSTATE.N/Z/C/V` — the same bits under another name |

That sharing is what lets a 64-bit kernel read a 32-bit application's arguments
straight out of `X0`–`X7`. Memory is shared too: a store from AArch32 is
immediately visible to AArch64.

**Exceptions leave AArch32 entirely.** An `SVC` at EL0 does not enter an AArch32
mode; it enters **EL1 in AArch64**, through the "lower EL using AArch32" vector
group at `VBAR + 0x600`, with `ESR.EC = 0x11` rather than `0x15` so the handler
can tell which execution state called it without reading the SPSR. `ERET`
returns, restoring the `T` bit so a T32 caller resumes in T32.

Verified against real GCC `-O2` AArch32 binaries: CRC-32 matching zlib, a
Fibonacci loop, and a word-copy loop all produce correct results at EL0.

What is not carried over: banking by AArch32 processor mode (at EL0 there is
only user mode), and the coprocessor space, so VFP and NEON reached through
`MRC`/`MCR` are undefined rather than mapped onto the AArch64 vector file.

## Multi-core

`Cluster` runs several cores sharing one memory hub, one exclusive monitor and
one interrupt controller:

```python
from armulator.armv8.cluster import Cluster

cluster = Cluster(4, memory_list, slice_size=8)
cluster.power_on(0, entry_point)
cluster.run(100_000)
```

Scheduling is cooperative round-robin. That is not how hardware interleaves, and
deliberately so: a deterministic order makes a failing test reproducible, where
true concurrency would not. `slice_size` is the knob — a smaller slice
interleaves more finely and is what to reach for when checking that a lock
actually holds.

**Secondary cores start parked** and are released through PSCI, the way real
firmware releases them, rather than by the harness reaching in and setting a PC.
`CPU_ON`, `CPU_OFF`, `AFFINITY_INFO` and `VERSION` are serviced on `SMC`,
standing in for the secure firmware that would answer at EL3. The context ID
arrives in `X0`.

The **exclusive monitor** is shared, with a 16-byte reservation granule (the
A57's). A successful `STXR` clears every core's reservation for the block; an
*ordinary* store clears other cores' reservations too, which is what stops a
lock released with `STR` from leaving a stale reservation that still looks
valid. Two variables in the same 16 bytes can make each other's exclusive stores
fail — real, occasionally surprising, and reproduced rather than smoothed over.

The GIC's CPU interface registers are banked per core, and SGIs work as
inter-processor interrupts, including broadcast to several cores at once.

## The memory model

Execution is in order, so by default every core sees every other core's writes
the instant they happen. That is sequential consistency — *stronger* than any
real machine — and it means lock-free code with missing barriers runs perfectly
here and then fails on silicon. The whole class of bug the barriers exist to
prevent is invisible.

Three models are available:

| Model | Behaviour |
|---|---|
| `SEQUENTIAL` | Default. Stores go straight to memory. |
| `RELAXED` | Stores are buffered and drained in program order. Exposes store/load reordering. |
| `ADVERSARIAL` | Stores drain in reverse order and loads may be moved in either direction. Exposes store/store, load/load and load/store reordering too. |

```python
from armulator.armv8.store_buffer import MemoryModel
cluster.set_memory_model(MemoryModel.ADVERSARIAL)
```

What this catches, using the three classic litmus tests:

```
Store buffering:  core0: x=1; r0=y      core1: y=1; r1=x
                  SC forbids r0==r1==0; every weak machine allows it.

Message passing:  writer: data=42; DMB; flag=1
                  reader: while(!flag){}; r=data
                  The reader has no barrier, so it can read data from
                  before the flag was ever set.

Load buffering:   core0: r0=x; y=1      core1: r1=y; x=1
                  SC forbids r0==r1==1; it needs each write to be seen
                  before the read that precedes it.
```

| | sequential | relaxed | adversarial |
|---|---|---|---|
| Store buffering, no barrier | `1,1` | **`0,0`** | **`0,0`** |
| Store buffering, with `DMB` | `1,1` | — | `1,1` |
| Message passing, plain reader | `42` | `42` | **`0`** |
| Message passing, reader `DMB`/`LDAR` | `42` | — | `42` |
| Load buffering, no barrier | `0,0` | `0,0` | **`1,1`** |
| Load buffering, with `DMB` | `0,0` | — | `0,0` |
| Spinlock, `STLR` release | 60/60 | 60/60 | 60/60 |
| Spinlock, plain `STR` release | 60/60 | — | **57/60** |

The `relaxed` column is the useful control on two of those rows. With a correct
writer, no *consistent* snapshot of memory shows the flag set without the data,
so store buffering provably cannot cause the message-passing failure; and
delaying stores can never make one visible *earlier*, so load buffering is out
of its reach by construction. Both need loads to move.

**How loads reorder.** A spin loop needs fresh reads to make progress; a
speculated load needs a stale one. A uniform lag gives one or the other, never
both. The rule used here is where a load sits relative to the last barrier: a
load whose address has already been read since then is a loop and reads current
memory, while a load reading somewhere new is answered *as of the barrier* — the
worst case the architecture permits.

**A load can move in either direction, but not both.** Message passing needs the
load performed *early*; load buffering needs it performed *late*. Those move it
opposite ways, and a deterministic model cannot do both to the same load. Which
one applies is decided by what actually follows it:

- a load with a **store** after it can be performed late, so the store becomes
  visible first — load/store reordering;
- a load with only **loads** after it is answered from before it executed —
  load/load reordering.

Load/store reordering is modelled by delaying the load rather than hoisting the
store, because hoisting would mean answering a load from the future, which a
forward simulation cannot do. Delaying is the same reordering seen from the
other side. A delayed load is settled the moment anything reads its register, so
the program never observes a provisional value — the reordering is only ever
visible when nothing depended on the result, which is exactly when a real
machine would have been free to reorder it too.

**What is never reordered.** Two accesses to the *same* address keep their
order, in the buffer and in the load rules alike. Coherence guarantees that
writes to one location are observed consistently, so reordering those would not
be weak memory behaviour but a broken machine — a core would watch its own
writes go backwards.

These are heuristics standing in for out-of-order execution, and they encode the
guarantee rather than the mechanism: without a barrier you have established
nothing about the order of two accesses, so the model assumes the worst.

Device memory is never buffered or reordered — a peripheral access has side
effects and must arrive in program order.

**This finds bugs; it does not prove their absence.** Detection depends on the
window, tunable with `latency`:

```python
cluster.set_memory_model(MemoryModel.ADVERSARIAL, latency=16)
```

The plain-`STR` release bug above is invisible at `latency=4` and caught at `8`.
The reordering happens either way; at 4 it lasts fewer instructions than the
polling loop takes to come round.

## What is not modelled

- **Widening and narrowing SIMD** (`SADDLV`, `SQXTN` and friends), table lookups
  (`TBL`/`TBX`), the remaining saturating forms such as `SQDMULH`, and FP16
  arithmetic.
- **AArch32 above EL0**, and the AArch32 coprocessor space — so VFP and NEON
  reached through `MRC`/`MCR` are undefined.
- **Fixed-point conversions**, and the debug, trace and performance-monitor
  registers beyond what firmware reads to identify the core.

All of these decode as undefined rather than executing incorrectly. The
distinction is not academic: while building the SIMD support, `USHR` — an
Advanced SIMD *scalar* instruction — was being swallowed by the scalar floating
point decoder and executed as an `FMADD`. It decoded, so a coverage count called
it covered, and a compiled function silently returned 6.5 instead of 11.0.
Decode rate is not correctness.
