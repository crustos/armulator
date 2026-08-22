"""
ARM GIC-400 Generic Interrupt Controller (GICv2).

This is the interrupt controller on the BCM2711 (Raspberry Pi 4) and, as
GIC-500, the architectural sibling on the Tegra X1.  It replaces the ad-hoc
"ask every device whether it is asserting" polling in :class:`Board` with
the real thing: prioritised, maskable, per-CPU-targeted interrupt routing.

The GIC splits into two register blocks:

    Distributor  (GICD, +0x1000)  global: enable, priority, target, pending
    CPU interface (GICC, +0x2000) per-core: acknowledge, priority mask, EOI

Interrupt IDs follow the GICv2 numbering::

    0-15     SGI   software generated (inter-processor)
    16-31    PPI   private peripheral (per-core timers etc.)
    32-1019  SPI   shared peripheral -- where real devices land

A device wired at SPI *n* uses interrupt ID ``32 + n``.

The handshake firmware performs is:

    1. read GICC_IAR      -> highest-priority pending ID, moves it to active
    2. service the device (which deasserts its line)
    3. write GICC_EOIR    -> deactivate, allowing the next interrupt

Reading IAR when nothing is pending returns the spurious ID 1023, which is
how interrupt handlers know to return without doing work.
"""

from armulator.peripherals.mmio import MMIODevice

SPURIOUS_ID = 1023
NUM_INTERRUPTS = 256        # plenty for the boards we model; GICv2 allows 1020
SGI_BASE = 0
PPI_BASE = 16
SPI_BASE = 32

# Distributor register offsets (relative to the GIC base, GICD at +0x1000)
GICD_BASE = 0x1000
GICD_CTLR = GICD_BASE + 0x000
GICD_TYPER = GICD_BASE + 0x004
GICD_IIDR = GICD_BASE + 0x008
GICD_IGROUPR = GICD_BASE + 0x080
GICD_ISENABLER = GICD_BASE + 0x100
GICD_ICENABLER = GICD_BASE + 0x180
GICD_ISPENDR = GICD_BASE + 0x200
GICD_ICPENDR = GICD_BASE + 0x280
GICD_ISACTIVER = GICD_BASE + 0x300
GICD_ICACTIVER = GICD_BASE + 0x380
GICD_IPRIORITYR = GICD_BASE + 0x400
GICD_ITARGETSR = GICD_BASE + 0x800
GICD_ICFGR = GICD_BASE + 0xC00
GICD_SGIR = GICD_BASE + 0xF00

# CPU interface register offsets (GICC at +0x2000)
GICC_BASE = 0x2000
GICC_CTLR = GICC_BASE + 0x00
GICC_PMR = GICC_BASE + 0x04
GICC_BPR = GICC_BASE + 0x08
GICC_IAR = GICC_BASE + 0x0C
GICC_EOIR = GICC_BASE + 0x10
GICC_RPR = GICC_BASE + 0x14
GICC_HPPIR = GICC_BASE + 0x18
GICC_IIDR = GICC_BASE + 0xFC


class Gic400(MMIODevice):
    """
    :param num_interrupts: highest interrupt ID modelled (default 256)
    :param num_cpus: number of CPU interfaces for targeting purposes

    Devices are attached with :meth:`connect`, which returns a callable the
    board uses to update that device's input line each step.
    """

    SIZE = 0x8000

    def __init__(self, num_interrupts=NUM_INTERRUPTS, num_cpus=4,
                 name='gic', trace=False):
        super().__init__(self.SIZE, name=name, trace=trace)
        self.num_interrupts = num_interrupts
        self.num_cpus = num_cpus

        self.enabled = [False] * num_interrupts
        self.pending = [False] * num_interrupts
        self.active = [False] * num_interrupts
        self.priority = [0xA0] * num_interrupts    # GICv2 reset: mid priority
        self.targets = [0x01] * num_interrupts     # default to CPU 0
        self.config = [0] * num_interrupts         # 0 = level, 1 = edge
        self.group = [0] * num_interrupts

        self.distributor_enabled = False
        # The CPU interface registers are banked per core on real hardware. An MMIO
        # access carries no requester identity, so the scheduler sets `current_cpu`
        # before stepping a core and accesses land in that core's bank.
        self.cpu_interface_enabled_per_cpu = [False] * num_cpus
        self.priority_mask_per_cpu = [0xFF] * num_cpus
        self.current_cpu = 0
        self.binary_point = 0

        #: Raw input line level per interrupt ID, before enable/priority.
        self.lines = [False] * num_interrupts
        #: Device name per interrupt ID, for readable traces.
        self.sources = {}
        #: Outstanding target mask per SGI, so an IPI can be delivered to several cores.
        self.sgi_targets = [0] * 16
        #: Active mask per SGI. The active state is banked per CPU interface for the
        #: banked interrupts, so one core servicing an IPI must not hide it from another.
        self.sgi_active = [0] * 16

    @property
    def cpu_interface_enabled(self):
        return self.cpu_interface_enabled_per_cpu[self.current_cpu]

    @cpu_interface_enabled.setter
    def cpu_interface_enabled(self, value):
        self.cpu_interface_enabled_per_cpu[self.current_cpu] = bool(value)

    @property
    def priority_mask(self):
        return self.priority_mask_per_cpu[self.current_cpu]

    @priority_mask.setter
    def priority_mask(self, value):
        self.priority_mask_per_cpu[self.current_cpu] = value

    def _is_active_for(self, intid: int, cpu) -> bool:
        """Whether ``intid`` is already being serviced by ``cpu``."""
        if intid < 16:
            if cpu is None:
                return bool(self.sgi_active[intid])
            return bool(self.sgi_active[intid] & (1 << cpu))
        return self.active[intid]

    def targets_cpu(self, intid: int, cpu: int) -> bool:
        """
        Whether ``intid`` is routed to ``cpu``.

        SGIs and the other banked interrupts below SPI_BASE are private to each core, so
        they are always considered targeted; SPIs consult the target byte the distributor
        holds for them.
        """
        if intid < 16:
            # An SGI is delivered to each targeted core separately, so it stays routed
            # to a core until that core has acknowledged it.
            return bool(self.sgi_targets[intid] & (1 << cpu))
        if intid < SPI_BASE:
            # PPIs are private to each core and always considered targeted.
            return True
        return bool(self.targets[intid] & (1 << cpu))

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def connect(self, device, spi: int, name=None):
        """
        Wire ``device``'s IRQ output to SPI number ``spi``.

        Returns the interrupt ID (``32 + spi``).  The board calls
        :meth:`refresh` each step to sample connected devices.
        """
        intid = SPI_BASE + spi
        if intid >= self.num_interrupts:
            raise ValueError(f'SPI {spi} exceeds modelled interrupt count')
        self.sources[intid] = (name or getattr(device, 'name', 'device'), device)
        return intid

    def refresh(self) -> None:
        """Sample every connected device's IRQ line into the distributor."""
        for intid, (_, device) in self.sources.items():
            self.set_line(intid, device.irq_pending)

    def set_line(self, intid: int, level: bool) -> None:
        """
        Drive interrupt ``intid``'s input line.

        Edge-triggered interrupts latch pending on the rising edge; level
        triggered ones track the line, so a device that keeps asserting
        stays pending until firmware clears the source.
        """
        level = bool(level)
        previous = self.lines[intid]
        self.lines[intid] = level
        if self.config[intid]:                     # edge triggered
            if level and not previous:
                self.pending[intid] = True
        else:
            # Level triggered: pending simply tracks the input line.  Active
            # and pending are independent states in GICv2, so a line going
            # low clears pending even while the interrupt is being serviced.
            self.pending[intid] = level
        self._update_output()

    # ------------------------------------------------------------------
    # Prioritisation
    # ------------------------------------------------------------------
    def _candidates(self, cpu=None):
        """
        Interrupt IDs that are pending, enabled and not already active.

        With ``cpu`` given, only interrupts routed to that core are considered.
        """
        return [
            i for i in range(self.num_interrupts)
            if self.pending[i] and self.enabled[i]
            and not self._is_active_for(i, cpu)
            and (cpu is None or self.targets_cpu(i, cpu))
        ]

    def highest_priority_pending(self, cpu=None):
        """
        The pending interrupt that ``cpu`` would acknowledge next, or ``None``.

        Lower priority *values* win; ties break toward the lower ID, which
        is what the architecture specifies. ``cpu`` defaults to the core the
        scheduler last selected.
        """
        if not self.distributor_enabled:
            return None
        if cpu is None:
            cpu = self.current_cpu
        candidates = self._candidates(cpu)
        if not candidates:
            return None
        best = min(candidates, key=lambda i: (self.priority[i], i))
        # GICC_PMR masks anything at or below the programmed priority.
        if self.priority[best] >= self.priority_mask_per_cpu[cpu]:
            return None
        return best

    def irq_pending_for(self, cpu: int) -> bool:
        """Whether the CPU interface is currently signalling an IRQ to ``cpu``."""
        return (self.cpu_interface_enabled_per_cpu[cpu]
                and self.highest_priority_pending(cpu) is not None)

    def _update_output(self):
        # `irq_pending` is the line as seen by the core the scheduler has selected.
        # Multi-core boards ask `irq_pending_for` per core instead.
        self.set_irq(self.irq_pending_for(self.current_cpu))

    # ------------------------------------------------------------------
    # Acknowledge / end-of-interrupt
    # ------------------------------------------------------------------
    def acknowledge(self) -> int:
        """
        Acknowledge the highest-priority pending interrupt (GICC_IAR read).

        Returns its ID, or :data:`SPURIOUS_ID` if nothing is pending.
        """
        intid = self.highest_priority_pending(self.current_cpu)
        if intid is None:
            return SPURIOUS_ID
        if intid < 16:
            self.sgi_active[intid] |= 1 << self.current_cpu
            # This core has taken its copy of the IPI; the interrupt stays pending for
            # any other core that was targeted and has not yet acknowledged.
            self.sgi_targets[intid] &= ~(1 << self.current_cpu)
            self.pending[intid] = bool(self.sgi_targets[intid])
            self._update_output()
            return intid
        self.active[intid] = True
        if self.config[intid]:
            # Edge triggered: the latch is consumed by acknowledging.
            self.pending[intid] = False
        else:
            # Level triggered: pending continues to reflect the line, so a
            # source still asserting will re-present once EOI deactivates it.
            self.pending[intid] = self.lines[intid]
        self._update_output()
        return intid

    def end_of_interrupt(self, intid: int) -> None:
        """Deactivate ``intid`` (GICC_EOIR write)."""
        intid &= 0x3FF
        if intid >= self.num_interrupts:
            return
        if intid < 16:
            self.sgi_active[intid] &= ~(1 << self.current_cpu)
            self._update_output()
            return
        self.active[intid] = False
        # A level-triggered source still asserting goes straight back to
        # pending -- the reason a handler must clear the device first.
        # Recomputed from the line rather than left stale, so a source that
        # deasserted during servicing does not re-fire spuriously.
        if not self.config[intid]:
            self.pending[intid] = self.lines[intid]
        self._update_output()

    def send_sgi(self, sgi_id: int, target_cpus: int = 0x01) -> None:
        """
        Raise a software generated interrupt (ID 0-15) - the inter-processor interrupt
        one core uses to poke another.
        """
        if not 0 <= sgi_id < 16:
            raise ValueError('SGI id must be 0-15')
        self.pending[sgi_id] = True
        self.targets[sgi_id] = target_cpus
        #: Which cores still owe an acknowledgement for this SGI.
        self.sgi_targets[sgi_id] = target_cpus
        self._update_output()

    # ------------------------------------------------------------------
    # Register naming
    # ------------------------------------------------------------------
    def register_name(self, offset):
        named = {
            GICD_CTLR: 'GICD_CTLR', GICD_TYPER: 'GICD_TYPER',
            GICD_IIDR: 'GICD_IIDR', GICD_SGIR: 'GICD_SGIR',
            GICC_CTLR: 'GICC_CTLR', GICC_PMR: 'GICC_PMR',
            GICC_BPR: 'GICC_BPR', GICC_IAR: 'GICC_IAR',
            GICC_EOIR: 'GICC_EOIR', GICC_RPR: 'GICC_RPR',
            GICC_HPPIR: 'GICC_HPPIR', GICC_IIDR: 'GICC_IIDR',
        }
        if offset in named:
            return named[offset]
        for base, label, stride in (
            (GICD_ISENABLER, 'GICD_ISENABLER', 32),
            (GICD_ICENABLER, 'GICD_ICENABLER', 32),
            (GICD_ISPENDR, 'GICD_ISPENDR', 32),
            (GICD_ICPENDR, 'GICD_ICPENDR', 32),
            (GICD_ISACTIVER, 'GICD_ISACTIVER', 32),
            (GICD_ICACTIVER, 'GICD_ICACTIVER', 32),
            (GICD_IGROUPR, 'GICD_IGROUPR', 32),
            (GICD_IPRIORITYR, 'GICD_IPRIORITYR', 4),
            (GICD_ITARGETSR, 'GICD_ITARGETSR', 4),
            (GICD_ICFGR, 'GICD_ICFGR', 16),
        ):
            span = (self.num_interrupts // stride) * 4
            if base <= offset < base + span:
                return f'{label}{(offset - base) // 4}'
        return f'+0x{offset:04X}'

    # ------------------------------------------------------------------
    # Bitmap helpers -- one bit per interrupt, 32 per register
    # ------------------------------------------------------------------
    def _read_bitmap(self, flags, index):
        base = index * 32
        return sum(
            1 << i for i in range(32)
            if base + i < self.num_interrupts and flags[base + i]
        )

    def _write_bitmap(self, flags, index, value, state):
        base = index * 32
        for i in range(32):
            if value & (1 << i) and base + i < self.num_interrupts:
                flags[base + i] = state

    def _read_bytes(self, values, index):
        base = index * 4
        return sum(
            (values[base + i] & 0xFF) << (8 * i)
            for i in range(4) if base + i < self.num_interrupts
        )

    def _write_bytes(self, values, index, value):
        base = index * 4
        for i in range(4):
            if base + i < self.num_interrupts:
                values[base + i] = (value >> (8 * i)) & 0xFF

    def _in_range(self, offset, base, stride):
        span = (self.num_interrupts // stride) * 4
        return base <= offset < base + span

    # ------------------------------------------------------------------
    # Register interface
    # ------------------------------------------------------------------
    def read_register(self, offset):
        if offset == GICD_CTLR:
            return int(self.distributor_enabled)
        if offset == GICD_TYPER:
            # ITLinesNumber field: (N+1)*32 interrupts supported.
            return ((self.num_interrupts // 32) - 1) | ((self.num_cpus - 1) << 5)
        if offset == GICD_IIDR:
            return 0x0200043B                       # ARM, GIC-400
        if offset == GICC_CTLR:
            return int(self.cpu_interface_enabled)
        if offset == GICC_PMR:
            return self.priority_mask
        if offset == GICC_BPR:
            return self.binary_point
        if offset == GICC_IAR:
            return self.acknowledge()
        if offset == GICC_HPPIR:
            pending = self.highest_priority_pending()
            return SPURIOUS_ID if pending is None else pending
        if offset == GICC_RPR:
            running = [i for i in range(self.num_interrupts) if self.active[i]]
            return min(self.priority[i] for i in running) if running else 0xFF
        if offset == GICC_IIDR:
            return 0x0202143B

        if self._in_range(offset, GICD_ISENABLER, 32):
            return self._read_bitmap(self.enabled, (offset - GICD_ISENABLER) // 4)
        if self._in_range(offset, GICD_ICENABLER, 32):
            return self._read_bitmap(self.enabled, (offset - GICD_ICENABLER) // 4)
        if self._in_range(offset, GICD_ISPENDR, 32):
            return self._read_bitmap(self.pending, (offset - GICD_ISPENDR) // 4)
        if self._in_range(offset, GICD_ICPENDR, 32):
            return self._read_bitmap(self.pending, (offset - GICD_ICPENDR) // 4)
        if self._in_range(offset, GICD_ISACTIVER, 32):
            return self._read_bitmap(self.active, (offset - GICD_ISACTIVER) // 4)
        if self._in_range(offset, GICD_ICACTIVER, 32):
            return self._read_bitmap(self.active, (offset - GICD_ICACTIVER) // 4)
        if self._in_range(offset, GICD_IGROUPR, 32):
            index = (offset - GICD_IGROUPR) // 4
            base = index * 32
            return sum(
                1 << i for i in range(32)
                if base + i < self.num_interrupts and self.group[base + i]
            )
        if self._in_range(offset, GICD_IPRIORITYR, 4):
            return self._read_bytes(self.priority, (offset - GICD_IPRIORITYR) // 4)
        if self._in_range(offset, GICD_ITARGETSR, 4):
            return self._read_bytes(self.targets, (offset - GICD_ITARGETSR) // 4)
        if self._in_range(offset, GICD_ICFGR, 16):
            index = (offset - GICD_ICFGR) // 4
            base = index * 16
            value = 0
            for i in range(16):
                if base + i < self.num_interrupts and self.config[base + i]:
                    value |= 0b10 << (2 * i)        # bit 1 of each pair
            return value
        return self.DEFAULT_READ

    def write_register(self, offset, value):
        if offset == GICD_CTLR:
            self.distributor_enabled = bool(value & 1)
            self._update_output()
            return
        if offset == GICC_CTLR:
            self.cpu_interface_enabled = bool(value & 1)
            self._update_output()
            return
        if offset == GICC_PMR:
            self.priority_mask = value & 0xFF
            self._update_output()
            return
        if offset == GICC_BPR:
            self.binary_point = value & 0x7
            return
        if offset == GICC_EOIR:
            self.end_of_interrupt(value)
            return
        if offset == GICD_SGIR:
            self.send_sgi(value & 0xF, (value >> 16) & 0xFF)
            return

        if self._in_range(offset, GICD_ISENABLER, 32):
            self._write_bitmap(self.enabled, (offset - GICD_ISENABLER) // 4, value, True)
        elif self._in_range(offset, GICD_ICENABLER, 32):
            self._write_bitmap(self.enabled, (offset - GICD_ICENABLER) // 4, value, False)
        elif self._in_range(offset, GICD_ISPENDR, 32):
            self._write_bitmap(self.pending, (offset - GICD_ISPENDR) // 4, value, True)
        elif self._in_range(offset, GICD_ICPENDR, 32):
            self._write_bitmap(self.pending, (offset - GICD_ICPENDR) // 4, value, False)
        elif self._in_range(offset, GICD_ISACTIVER, 32):
            self._write_bitmap(self.active, (offset - GICD_ISACTIVER) // 4, value, True)
        elif self._in_range(offset, GICD_ICACTIVER, 32):
            self._write_bitmap(self.active, (offset - GICD_ICACTIVER) // 4, value, False)
        elif self._in_range(offset, GICD_IGROUPR, 32):
            index = (offset - GICD_IGROUPR) // 4
            base = index * 32
            for i in range(32):
                if base + i < self.num_interrupts:
                    self.group[base + i] = (value >> i) & 1
        elif self._in_range(offset, GICD_IPRIORITYR, 4):
            self._write_bytes(self.priority, (offset - GICD_IPRIORITYR) // 4, value)
        elif self._in_range(offset, GICD_ITARGETSR, 4):
            self._write_bytes(self.targets, (offset - GICD_ITARGETSR) // 4, value)
        elif self._in_range(offset, GICD_ICFGR, 16):
            index = (offset - GICD_ICFGR) // 4
            base = index * 16
            for i in range(16):
                if base + i < self.num_interrupts:
                    self.config[base + i] = (value >> (2 * i + 1)) & 1
        else:
            return
        self._update_output()
