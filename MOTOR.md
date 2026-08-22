# Motor control

Firmware writes I2C registers; a shaft turns a measurable distance. Everything between
those two points is modelled, so a test can assert on the outcome rather than on the
register writes that were supposed to cause it.

```
   ARM firmware
        │  writes registers
   BCM2711 I2C controller          armulator.peripherals.serial_bus
        │  I2C transaction
   PCA9685 PWM controller          armulator.peripherals.pca9685
        │  16 PWM channels
   TB6612FNG H-bridges             armulator.peripherals.motor
        │  signed drive
   DC motor / stepper              armulator.peripherals.motor
        │
   position, in revolutions or steps
```

This is the Adafruit DC and Stepper Motor HAT: a PCA9685 driving four H-bridges. The same
pieces work for any board built the same way, which is most of them.

For the general API see [README.md](README.md); for the Raspberry Pi boards these
attach to, [RASPI.md](RASPI.md).

## Contents

- [The shortest useful example](#the-shortest-useful-example)
- [Why the layers are separate](#why-the-layers-are-separate)
- [Driving from firmware](#driving-from-firmware)
- [The PCA9685, and three ways to get it wrong](#the-pca9685-and-three-ways-to-get-it-wrong)
- [H-bridges: coast is not brake](#h-bridges-coast-is-not-brake)
- [DC motors and the time problem](#dc-motors-and-the-time-problem)
- [Steppers and missed steps](#steppers-and-missed-steps)
- [The channel map](#the-channel-map)
- [What is not modelled](#what-is-not-modelled)

## The shortest useful example

```python
from armulator.boards import RaspberryPi4
from armulator.peripherals import MotorHat

board = RaspberryPi4()
hat = MotorHat().attach_to(board)      # PCA9685 lands on the Pi's I2C bus at 0x60
motor = hat.attach_dc_motor(1)         # a DC motor on the M1 terminals

# ... firmware runs and programs the controller ...

hat.advance(1.0)                       # one second of shaft time
assert motor.position > 0              # it turned, and it turned forwards
```

`MotorHat` is not a `Board` subclass, because a HAT is not a board — it plugs onto one.
`attach_to` puts the controller on the Pi's I2C bus and wires the H-bridges to the PWM
channels the HAT routes to them.

## Why the layers are separate

The PCA9685 knows nothing about motors. It produces sixteen PWM outputs; what they are
wired to is the board designer's business. The H-bridge knows nothing about I2C. The
motor knows nothing about either.

Keeping them apart is what makes the stack assertable. If the PWM controller had motor
logic baked into it, a test could only ever check that the right registers were written —
which is a claim about the code you already have, not about what it does. Split up, the
test can say *the shaft turned this far in this direction*, and that claim survives a
rewrite of the driver.

It also means the pieces recombine. The same PCA9685 model drives a servo HAT; the same
H-bridge model works behind a GPIO-driven L298N with no PWM controller at all.

## Driving from firmware

The end-to-end path, with real ARM code on the Pi. Writing a PCA9685 register means an
I2C transaction: set the slave address once, then for each write fill the FIFO with the
register number followed by the data and strobe ST.

```python
from armulator.boards import RaspberryPi4
from armulator.boards.firmware import firmware
from armulator.peripherals import MotorHat
from armulator.peripherals.pca9685 import LED0_ON_L, MODE1, MODE1_AI, MODE1_SLEEP, PRE_SCALE

I2C = 0xFE804000
BSC_C, BSC_DLEN, BSC_A, BSC_FIFO = 0x00, 0x08, 0x0C, 0x10
C_ST, C_CLEAR, C_I2CEN = 1 << 7, 0b11 << 4, 1 << 15


def i2c_write(register, *values):
    payload = ''.join(f'    mov r2, #{v}\n    str r2, [r0, #{BSC_FIFO}]\n'
                      for v in (register,) + values)
    return (f'    mov r2, #{C_CLEAR}\n    str r2, [r0, #{BSC_C}]\n'
            f'    mov r2, #{1 + len(values)}\n    str r2, [r0, #{BSC_DLEN}]\n'
            f'{payload}'
            f'    ldr r2, ={C_I2CEN | C_ST}\n    str r2, [r0, #{BSC_C}]\n')


board = RaspberryPi4()
hat = MotorHat().attach_to(board)
motor = hat.attach_dc_motor(1)

source = (
    f'    ldr r0, ={I2C}\n'
    f'    mov r1, #0x60\n'
    f'    str r1, [r0, #{BSC_A}]\n'
    + i2c_write(MODE1, MODE1_SLEEP)            # sleep, so the prescaler is writable
    + i2c_write(PRE_SCALE, 3)                  # ~1.5 kHz
    + i2c_write(MODE1, MODE1_AI)               # wake, auto-increment on
    + i2c_write(LED0_ON_L + 4 * 10, 0, 0, 0xFF, 0x0F)   # IN1 full on
    + i2c_write(LED0_ON_L + 4 * 9,  0, 0, 0, 0)         # IN2 off
    + i2c_write(LED0_ON_L + 4 * 8,  0, 0, 0xFF, 0x07)   # PWM at half
)

board.load(board.CODE_BASE, firmware(source, address=board.CODE_BASE))
board.start()
board.run(5000)

hat.advance(1.0)
print(hat.format_state())
```

```
<Pca9685 0x60 1526Hz>
  M1: forward  duty=0.50  pos=+0.850rev speed=+0.998rev/s
  M2: coast    duty=0.00
  M3: coast    duty=0.00
  M4: coast    duty=0.00
```

Note the frequency: the firmware asked for something near 1600 Hz and got 1526. That is
not an emulation artefact — see below.

## The PCA9685, and three ways to get it wrong

The register model is faithful because that is where driver bugs start. Three behaviours
in particular are reproduced rather than smoothed over, because each produces a driver
that looks correct and does nothing.

### The chip wakes up asleep

`MODE1.SLEEP` is set at reset, and a sleeping PCA9685 drives nothing whatever its LED
registers say. Firmware that programs the outputs and never clears `SLEEP` gets silence:

```python
hat.controller.sleeping        # True at reset
hat.controller.duty_cycle(8)   # 0.0, whatever was written
```

### The prescaler is only writable while asleep

Setting the PWM frequency requires putting the chip *back* to sleep first. A write while
awake is ignored — no error, no effect:

```python
pwm.write([PRE_SCALE, 0x63])   # awake: ignored
pwm.prescale                   # unchanged

pwm.write([MODE1, MODE1_AI | MODE1_SLEEP])
pwm.write([PRE_SCALE, 0x63])   # asleep: accepted
pwm.write([MODE1, MODE1_AI])
pwm.frequency                  # 61.0
```

And the frequency you get is not the one you asked for. It is
`25 MHz / (4096 x (prescale + 1))`, which cannot land on every value: ask for 1600 Hz and
the nearest prescaler gives 1526 Hz. `prescale_for()` does the arithmetic, but the
rounding is real and a driver that reads back and compares will be surprised.

### Auto-increment is off at reset

A driver that writes all four LED registers in one transaction, without setting
`MODE1.AI`, writes four values into the *same* register:

```python
pwm.write([MODE1, 0])                            # AI off
pwm.write([LED0_ON_L, 0x11, 0x22, 0x33, 0x44])
pwm.registers[LED0_ON_L]                         # 0x44 — the last one wins
pwm.registers[LED0_ON_L + 1]                     # 0 — never written
```

One more, less common but worth knowing: bit 4 of `ON_H` and `OFF_H` force a channel
fully on or fully off, and **full-on wins if both are set**. Most people guess the other
way round, and the result is an output stuck high.

## H-bridges: coast is not brake

An H-bridge takes two direction inputs and a PWM enable:

| IN1 | IN2 | Result |
|---|---|---|
| 0 | 0 | **coast** — outputs floating, the motor freewheels |
| 1 | 0 | forward |
| 0 | 1 | reverse |
| 1 | 1 | **brake** — outputs shorted, the motor stops fast |

The difference between coast and brake is real and is modelled. A braked motor is shorted
through the bridge and stops sharply; a coasting one spins down under friction alone:

```python
# from 2.0 rev/s, after 0.2 seconds
coast  ->  1.28 rev/s
brake  ->  0.01 rev/s
```

Driver code that means to stop a motor and only coasts is a common bug, and one that is
invisible if the model treats both as "not driving".

## DC motors and the time problem

The emulator counts instructions, not seconds. The relationship between the two depends
on a clock speed nothing here models, so **time is explicit**:

```python
board.run(2000)      # firmware programs the controller
hat.advance(0.5)     # half a second of shaft time
```

Pretending instruction counts were seconds would produce numbers that looked precise and
meant nothing. Making the caller say how much time passed is honest about what is being
simulated.

The motor model is first order: drive produces torque, friction opposes motion, speed
settles at a terminal value proportional to drive. That answers the questions a driver
test actually asks — did it turn, which way, roughly how far, did it stop when told —
without pretending to a fidelity nobody has the parameters for. Parameters that matter
are adjustable:

```python
motor = hat.attach_dc_motor(1, free_speed=3.0, spin_up=0.1, stall_drive=0.05)
```

The integration is closed-form rather than stepped, so `advance(1.0)` and a hundred calls
to `advance(0.01)` give the same answer. Tests are not sensitive to step size.

`stall_drive` is the duty below which static friction wins and nothing moves — which real
motors do, and which catches a driver that sets a duty cycle too low to be useful:

```python
motor.stalled     # True: being driven, but not hard enough to turn
```

## Steppers and missed steps

A bipolar stepper needs two H-bridges, one per coil, so it occupies two motor positions:
stepper 1 is M1 and M2, stepper 2 is M3 and M4.

```python
stepper = hat.attach_stepper(1, steps_per_revolution=200)
```

Unlike a DC motor there is no speed to settle — the rotor follows the field the coils
produce and moves in discrete steps as that field rotates. So a stepper has no
`advance()`: it moves when the coils change, which has already happened by the time you
would call it.

The failure worth catching is a **missed step**. If firmware advances the field by more
than one position — stepping too fast, or getting the sequence wrong — a real rotor cannot
follow and slips. That is counted rather than silently applied:

```python
stepper.steps          # where the rotor actually is
stepper.missed_steps   # how far the commanded position has drifted from it
```

A test that asserts `missed_steps == 0` alongside the expected `steps` is checking
something a register-level test cannot see at all.

One detail: the first energisation snaps the rotor to whatever the field says without
counting as a step, so driving a 200-position sequence from cold gives 199 steps. That is
the physical behaviour, not an off-by-one.

## The channel map

Which PWM channel drives which H-bridge input comes from the HAT's schematic. Nothing in
the chip or the driver implies it, and getting it wrong is the classic mistake when
writing a driver from scratch:

| Motor | PWM | IN2 | IN1 |
|---|---|---|---|
| M1 | 8 | 9 | 10 |
| M2 | 13 | 12 | 11 |
| M3 | 2 | 3 | 4 |
| M4 | 7 | 6 | 5 |

**M2 and M4 have their direction channels in the opposite order to M1 and M3.** That is
not a typo — it falls out of the PCB routing. A driver that assumes a uniform pattern
drives two of the four motors backwards, and the model reproduces that faithfully rather
than being forgiving about it.

The map is exported as `MOTOR_CHANNELS` if you need it, and overridable if your board
routes differently.

## What is not modelled

- **Current, voltage and temperature.** There is no supply rail, no current limit, no
  thermal shutdown. `stall_drive` is a friction threshold, not a current measurement.
- **Back-EMF and encoder feedback.** Position is known exactly because the model computes
  it; there is no encoder to read and no quadrature signal to decode. Closed-loop control
  code can be driven, but it has nothing to close the loop on.
- **Microstepping.** The stepper model has four full-step field positions. Half-stepping
  energises both coils at once, which the field model reports as ambiguous and ignores.
- **The TB6612FNG's STBY pin**, and shoot-through protection. The bridge truth table is
  modelled; the analogue failure modes are not.
- **I2C bus errors** beyond the NACK the controller already models — no clock stretching,
  arbitration loss, or bus lockup.

The motor dynamics are a first-order approximation with no claim to matching a specific
part. If you need to know whether your motor will actually reach a speed, measure the
motor. What this is for is checking that your driver writes the right registers in the
right order and that the result moves the right way.
