"""
Recording emulated hardware state over time, and plotting it.

:mod:`armulator.telemetry.recorder` is pure Python and always importable.
:mod:`armulator.telemetry.plot` needs matplotlib and is deliberately *not* imported
here, so that ``from armulator.telemetry import MotorTelemetry`` works on a machine
without it. Import the plotting helpers explicitly when you want them::

    from armulator.telemetry.plot import plot_motor
"""

from armulator.telemetry.recorder import (
    BRIDGE_FIELDS, DC_FIELDS, MODE_CODES, STEPPER_FIELDS, MotorTelemetry,
)

__all__ = [
    'MotorTelemetry',
    'MODE_CODES', 'BRIDGE_FIELDS', 'DC_FIELDS', 'STEPPER_FIELDS',
]
