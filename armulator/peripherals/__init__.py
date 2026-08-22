from armulator.peripherals.gic400 import Gic400
from armulator.peripherals.gpio_bcm import BcmGpio, GpioFunction, Pull
from armulator.peripherals.gpio_tegra import TegraGpio
from armulator.peripherals.mmio import Access, MMIODevice, UnimplementedDevice
from armulator.peripherals.motor import (
    BRAKE, COAST, FORWARD, REVERSE, DcMotor, HBridge, StepperMotor,
)
from armulator.peripherals.motor_hat import MOTOR_CHANNELS, MotorChannel, MotorHat
from armulator.peripherals.pca9685 import Pca9685
from armulator.peripherals.serial_bus import (
    Bcm2835I2c, Bcm2835Spi, I2cSlaveDevice, SpiLoopback, SpiSlaveDevice,
)
from armulator.peripherals.spi_slave import Bcm2835SpiSlave, address_octet
from armulator.peripherals.spi_tegra import Tegra210Spi
from armulator.peripherals.uart_8250 import TegraUart, Uart8250
from armulator.peripherals.uart_pl011 import BcmSystemTimer, Pl011Uart

__all__ = [
    'MMIODevice', 'UnimplementedDevice', 'Access',
    'BcmGpio', 'GpioFunction', 'Pull', 'TegraGpio',
    'Pl011Uart', 'BcmSystemTimer', 'Gic400',
    'Uart8250', 'TegraUart',
    'Bcm2835Spi', 'Bcm2835I2c', 'SpiSlaveDevice', 'SpiLoopback',
    'I2cSlaveDevice', 'Bcm2835SpiSlave', 'address_octet',
    'Tegra210Spi',
    'Pca9685', 'MotorHat', 'MotorChannel', 'MOTOR_CHANNELS',
    'HBridge', 'DcMotor', 'StepperMotor',
    'FORWARD', 'REVERSE', 'BRAKE', 'COAST',
]
