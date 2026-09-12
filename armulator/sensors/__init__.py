"""
Sensor models: peripherals that measure the world rather than act on it.

The world itself lives in :mod:`armulator.sensors.geometry`, so that more than one
sensor can range against the same room.
"""

from armulator.sensors.geometry import Obstacle, Room
from armulator.sensors.lidar import (
    CMD_GET_HEALTH, CMD_GET_INFO, CMD_RESET, CMD_SCAN, CMD_STOP, SYNC,
    SYNC_RESPONSE, Lidar, Measurement, decode_measurement, encode_descriptor,
    encode_measurement,
)

__all__ = [
    'Room', 'Obstacle',
    'Lidar', 'Measurement',
    'encode_measurement', 'decode_measurement', 'encode_descriptor',
    'SYNC', 'SYNC_RESPONSE',
    'CMD_SCAN', 'CMD_STOP', 'CMD_RESET', 'CMD_GET_INFO', 'CMD_GET_HEALTH',
]
