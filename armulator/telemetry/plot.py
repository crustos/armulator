"""
Plotting recorded motor telemetry with matplotlib.

matplotlib is an optional dependency. Nothing in :mod:`armulator.telemetry.recorder`
imports it, and nothing here is imported at package level, so the emulator and its test
suite run without it installed. Import failures are turned into a message that says what
to install rather than a bare :class:`ImportError` from somewhere deep in a helper.

Every function takes a :class:`~armulator.telemetry.recorder.MotorTelemetry` and returns
the matplotlib axes or figure it drew on, so a caller can keep styling, add annotations,
save or show. Nothing here calls ``show()`` -- that is the caller's decision, and calling
it in a headless test run hangs.

WHY THE PLOTS LOOK LIKE THEY DO
-------------------------------
Duty cycle and mode are piecewise constant: they hold a value until firmware writes a
register, then jump. Drawing them as sloped lines between samples would suggest a ramp
the hardware never produced, so they are drawn with ``steps-post``. Speed and position
are continuous and drawn as ordinary lines.
"""

from armulator.telemetry.recorder import MODE_CODES

#: Signals that hold their value between writes and should be drawn as steps.
_STEP_SIGNALS = {'duty', 'drive', 'mode_code', 'braking', 'stalled', 'steps', 'missed_steps'}

_AXIS_LABELS = {
    'duty': 'duty cycle',
    'drive': 'signed drive',
    'mode_code': 'bridge mode',
    'braking': 'braking',
    'position': 'position (rev)',
    'speed': 'speed (rev/s)',
    'angle': 'angle (deg)',
    'stalled': 'stalled',
    'steps': 'steps',
    'missed_steps': 'missed steps',
}


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'plotting telemetry needs matplotlib, which armulator does not install by '
            'default. Install it with:  python3 -m pip install armulator[plot]'
        ) from error
    return plt


def _channel_label(telemetry, index):
    load = telemetry.hat.channels[index].load
    position = f'M{index}'
    # attach_dc_motor names the load after its position, so the two usually agree and
    # repeating it reads as a mistake. Only qualify when the name adds something.
    if load is None or load.name == position:
        return position
    return f'{position} ({load.name})'


def plot_signal(telemetry, field, channels=None, ax=None, **kwargs):
    """
    Plot one signal for one or more motor positions on a single axes.

    :param field: signal name, e.g. ``'duty'``, ``'speed'``, ``'position'``
    :param channels: positions to draw; every recorded position that has the signal
    :param ax: draw here instead of creating a figure
    :returns: the axes drawn on
    """
    plt = _pyplot()
    ax = ax or plt.subplots()[1]

    if channels is None:
        channels = [i for i in telemetry.channels if field in telemetry.data[i]]
    if not channels:
        raise ValueError(f'no recorded motor position has a {field!r} signal')

    style = {'drawstyle': 'steps-post'} if field in _STEP_SIGNALS else {}
    style.update(kwargs)

    for index in channels:
        times, values = telemetry.series(index, field)
        ax.plot(times, values, label=_channel_label(telemetry, index), **style)

    ax.set_xlabel('time (s)')
    ax.set_ylabel(_AXIS_LABELS.get(field, field))
    ax.grid(True, alpha=0.3)
    if len(channels) > 1:
        ax.legend(loc='best', fontsize='small')

    if field == 'mode_code':
        # Raw numbers on this axis mean nothing; label them with the modes they stand
        # for, and give the line room so brake at the top is not clipped.
        ticks = sorted(MODE_CODES.values())
        names = {code: name for name, code in MODE_CODES.items()}
        ax.set_yticks(ticks)
        ax.set_yticklabels([names[code] for code in ticks])
        ax.set_ylim(min(ticks) - 0.4, max(ticks) + 0.4)
    elif field in ('braking', 'stalled'):
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['no', 'yes'])
        ax.set_ylim(-0.2, 1.2)

    return ax


def plot_duty(telemetry, channels=None, ax=None, signed=True, **kwargs):
    """
    Plot what the bridges are driving.

    ``signed=True`` plots :attr:`drive`, which carries direction -- reverse shows below
    zero. ``signed=False`` plots raw duty cycle, which is what a driver wrote and is
    always positive. The signed version is usually the more useful of the two, because
    a duty plot alone cannot distinguish forward from reverse.
    """
    field = 'drive' if signed else 'duty'
    ax = plot_signal(telemetry, field, channels=channels, ax=ax, **kwargs)
    if signed:
        ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.4)
    return ax


def plot_motor(telemetry, index, fields=('drive', 'speed', 'position'), figsize=None):
    """
    A stacked plot of one motor: several signals sharing a time axis.

    Sharing the x axis is the point -- it lines up a register write with the speed
    change it caused, which is the thing you are usually looking for.

    :returns: ``(figure, axes_list)``
    """
    plt = _pyplot()
    available = telemetry.data[index]
    fields = [field for field in fields if field in available]
    if not fields:
        raise ValueError(
            f'M{index} recorded none of those signals; it has: '
            f'{", ".join(telemetry.fields(index))}'
        )

    figure, axes = plt.subplots(
        len(fields), 1, sharex=True, figsize=figsize or (9, 2.2 * len(fields))
    )
    if len(fields) == 1:
        axes = [axes]

    for ax, field in zip(axes, fields):
        plot_signal(telemetry, field, channels=[index], ax=ax)
        ax.set_xlabel('')
        if field == 'drive':
            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.4)
    axes[-1].set_xlabel('time (s)')
    figure.suptitle(_channel_label(telemetry, index))
    figure.tight_layout()
    return figure, list(axes)


def plot_hat(telemetry, fields=('drive', 'speed'), channels=None, figsize=None):
    """
    Every active motor position on the HAT, one column each.

    Defaults to the positions that actually did something, since a HAT with one motor
    on it would otherwise produce three columns of flat lines.

    :returns: ``(figure, axes_grid)``
    """
    plt = _pyplot()
    if channels is None:
        channels = telemetry.active_channels or telemetry.channels
    channels = list(channels)

    rows = len(fields)
    figure, grid = plt.subplots(
        rows, len(channels), sharex=True, squeeze=False,
        figsize=figsize or (4.5 * len(channels), 2.2 * rows),
    )

    for column, index in enumerate(channels):
        for row, field in enumerate(fields):
            ax = grid[row][column]
            if field not in telemetry.data[index]:
                ax.set_axis_off()
                continue
            plot_signal(telemetry, field, channels=[index], ax=ax)
            ax.set_xlabel('')
            if column > 0:
                ax.set_ylabel('')
            if row == 0:
                ax.set_title(_channel_label(telemetry, index), fontsize='medium')
        grid[-1][column].set_xlabel('time (s)')

    figure.tight_layout()
    return figure, grid
