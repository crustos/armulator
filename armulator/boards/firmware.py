"""
Helpers for turning ARM assembly into bytes for the board models.

Uses Keystone if it is installed.  Keystone is only needed to *write* test
firmware -- the board and peripheral models have no dependency on it, so
pre-assembled byte strings work fine without it.
"""

try:
    from keystone import (
        KS_ARCH_ARM,
        KS_ARCH_ARM64,
        KS_MODE_ARM,
        KS_MODE_LITTLE_ENDIAN,
        KS_MODE_THUMB,
        Ks,
    )
    HAVE_KEYSTONE = True
except ImportError:                                    # pragma: no cover
    HAVE_KEYSTONE = False


def _require_keystone():
    if not HAVE_KEYSTONE:
        raise RuntimeError(
            'keystone-engine is required to assemble source; '
            'install it with `pip install keystone-engine`, or pass '
            'pre-assembled bytes to Board.load()'
        )


def assemble(source: str, thumb: bool = False, address: int = 0) -> bytes:
    """
    Assemble ARM (or Thumb) ``source`` into bytes.

    :param source: assembly text, newline or semicolon separated
    :param thumb: assemble as T32 rather than A32
    :param address: base address, so PC-relative fixups come out right
    """
    _require_keystone()
    ks = Ks(KS_ARCH_ARM, KS_MODE_THUMB if thumb else KS_MODE_ARM)
    encoding, _ = ks.asm(source, address)
    return bytes(encoding)


def assemble_a64(source: str, address: int = 0) -> bytes:
    """
    Assemble AArch64 ``source`` into bytes.

    :param source: assembly text, newline or semicolon separated
    :param address: base address, so PC-relative fixups come out right

    Keystone is off by one when an ADR or branch operand is written as a bare number
    (``adr x0, #0`` encodes an offset of -1).  Write firmware with labels rather than
    numeric offsets and the encodings come out right.
    """
    _require_keystone()
    ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
    encoding, _ = ks.asm(source, address)
    return bytes(encoding)


#: A tight self-branch.  Bare-metal firmware ends with this, and
#: ``Board.run`` detects it and stops.  Valid in A32, T32 and A64 alike.
HALT = 'b .'


def firmware(body: str, thumb: bool = False, address: int = 0) -> bytes:
    """Assemble ``body`` with a halt loop appended."""
    return assemble(body + '\n' + HALT, thumb=thumb, address=address)


def firmware_a64(body: str, address: int = 0) -> bytes:
    """Assemble AArch64 ``body`` with a halt loop appended."""
    return assemble_a64(body + '\n' + HALT, address=address)
