"""
Helpers for turning ARM assembly into bytes for the board models.

Uses Keystone if it is installed.  Keystone is only needed to *write* test
firmware -- the board and peripheral models have no dependency on it, so
pre-assembled byte strings work fine without it.
"""

try:
    from keystone import KS_ARCH_ARM, KS_MODE_ARM, KS_MODE_THUMB, Ks
    HAVE_KEYSTONE = True
except ImportError:                                    # pragma: no cover
    HAVE_KEYSTONE = False


def assemble(source: str, thumb: bool = False, address: int = 0) -> bytes:
    """
    Assemble ARM (or Thumb) ``source`` into bytes.

    :param source: assembly text, newline or semicolon separated
    :param thumb: assemble as T32 rather than A32
    :param address: base address, so PC-relative fixups come out right
    """
    if not HAVE_KEYSTONE:
        raise RuntimeError(
            'keystone-engine is required to assemble source; '
            'install it with `pip install keystone-engine`, or pass '
            'pre-assembled bytes to Board.load()'
        )
    ks = Ks(KS_ARCH_ARM, KS_MODE_THUMB if thumb else KS_MODE_ARM)
    encoding, _ = ks.asm(source, address)
    return bytes(encoding)


#: A tight self-branch.  Bare-metal firmware ends with this, and
#: ``Board.run`` detects it and stops.
HALT = 'b .'


def firmware(body: str, thumb: bool = False, address: int = 0) -> bytes:
    """Assemble ``body`` with a halt loop appended."""
    return assemble(body + '\n' + HALT, thumb=thumb, address=address)
