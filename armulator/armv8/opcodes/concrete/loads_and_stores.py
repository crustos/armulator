"""
Concrete encodings for the load and store instructions.

The size and opc fields interact in a way that is worth stating once rather than
repeating in every encoding: opc<1> selects between "store or zero-extending load" and
"sign-extending load or prefetch", and the combination decides both the access width and
the width of the destination register. Two combinations are reserved: a sign-extending
load of a doubleword (there is nothing wider to extend into) and LDRSW into a 32-bit
register (which would discard the sign extension it just performed).
"""

from armulator.armv8.bits_ops import bit_at, sign_extend, substring
from armulator.armv8.enums import ExtendType, MemOp
from armulator.armv8.opcodes.abstract_opcodes.load_literal import LoadLiteral
from armulator.armv8.opcodes.abstract_opcodes.load_store_exclusive import LoadStoreExclusive
from armulator.armv8.opcodes.abstract_opcodes.load_store_immediate import LoadStoreImmediate
from armulator.armv8.opcodes.abstract_opcodes.load_store_pair import LoadStorePair
from armulator.armv8.opcodes.abstract_opcodes.load_store_register_offset import (
    LoadStoreRegisterOffset,
)


def decode_size_opc(size, opc):
    """
    Resolve (size, opc) into (memop, signed, regsize), or None when reserved.
    """
    if not bit_at(opc, 1):
        # Store, or a load that zero extends.
        memop = MemOp.LOAD if bit_at(opc, 0) else MemOp.STORE
        regsize = 64 if size == 0b11 else 32
        return memop, False, regsize
    if size == 0b11:
        if bit_at(opc, 0):
            return None
        return MemOp.PREFETCH, False, 64
    if size == 0b10 and bit_at(opc, 0):
        # LDRSW into a 32-bit register is reserved.
        return None
    return MemOp.LOAD, True, (32 if bit_at(opc, 0) else 64)


class LoadStoreUnsignedOffsetA64(LoadStoreImmediate):
    """
    size 1 1 1 V 0 1 opc imm12 Rn Rt - the scaled, unsigned immediate offset form.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = substring(instr, 31, 30)
        opc = substring(instr, 23, 22)
        decoded = decode_size_opc(size, opc)
        if decoded is None:
            return None
        memop, signed, regsize = decoded
        # The immediate counts elements, so it scales with the access width.
        offset = substring(instr, 21, 10) << size
        return LoadStoreUnsignedOffsetA64(
            instr, t=substring(instr, 4, 0), n=substring(instr, 9, 5), offset=offset,
            memop=memop, signed=signed, regsize=regsize, datasize=8 << size,
            wback=False, postindex=False,
        )


class LoadStoreIndexedA64(LoadStoreImmediate):
    """
    size 1 1 1 V 0 0 opc 0 imm9 op4 Rn Rt

    op4 picks the addressing mode: 00 unscaled (LDUR/STUR), 01 post-indexed,
    10 unprivileged (LDTR/STTR), 11 pre-indexed.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = substring(instr, 31, 30)
        opc = substring(instr, 23, 22)
        decoded = decode_size_opc(size, opc)
        if decoded is None:
            return None
        memop, signed, regsize = decoded
        op4 = substring(instr, 11, 10)
        offset = sign_extend(substring(instr, 20, 12), 9, 64)

        wback = op4 in (0b01, 0b11)
        postindex = op4 == 0b01
        n = substring(instr, 9, 5)
        t = substring(instr, 4, 0)
        # Writing back to the register being loaded leaves the result unpredictable.
        if wback and n != 31 and n == t and memop == MemOp.LOAD:
            return None

        return LoadStoreIndexedA64(
            instr, t=t, n=n, offset=offset, memop=memop, signed=signed,
            regsize=regsize, datasize=8 << size, wback=wback, postindex=postindex,
        )


class LoadStoreRegisterOffsetA64(LoadStoreRegisterOffset):
    """
    size 1 1 1 V 0 0 opc 1 Rm option S 1 0 Rn Rt
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = substring(instr, 31, 30)
        opc = substring(instr, 23, 22)
        decoded = decode_size_opc(size, opc)
        if decoded is None:
            return None
        memop, signed, regsize = decoded
        option = substring(instr, 15, 13)
        # option<1> must be set: a plain 32-bit index with no extension has no encoding.
        if not bit_at(option, 1):
            return None
        scaled = bit_at(instr, 12)
        return LoadStoreRegisterOffsetA64(
            instr, t=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), extend_type=ExtendType(option),
            shift=size if scaled else 0, memop=memop, signed=signed,
            regsize=regsize, datasize=8 << size,
        )


class LoadStorePairA64(LoadStorePair):
    """
    opc 1 0 1 V idx L imm7 Rt2 Rn Rt

    idx at instr[24:23] picks the addressing mode: 00 non-temporal, 01 post-indexed,
    10 plain offset, 11 pre-indexed.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        opc = substring(instr, 31, 30)
        if opc == 0b11:
            return None
        load = bit_at(instr, 22)
        # opc 01 is LDPSW on load; the store form is a tagged pair, unallocated here.
        signed = opc == 0b01
        if signed and not load:
            return None

        # LDPSW moves words, so it scales as a 32-bit access despite its 64-bit result.
        scale = 3 if opc == 0b10 else 2
        datasize = 8 << scale
        offset = sign_extend(substring(instr, 21, 15), 7, 64) << scale

        idx = substring(instr, 24, 23)
        if idx == 0b00:
            wback, postindex = False, False       # non-temporal
        elif idx == 0b01:
            wback, postindex = True, True
        elif idx == 0b10:
            wback, postindex = False, False
        else:
            wback, postindex = True, False

        t = substring(instr, 4, 0)
        t2 = substring(instr, 14, 10)
        n = substring(instr, 9, 5)
        if load and t == t2:
            # Loading the same register twice leaves it unpredictable.
            return None
        if wback and n != 31 and n in (t, t2) and load:
            return None

        return LoadStorePairA64(
            instr, t=t, t2=t2, n=n, offset=offset,
            memop=MemOp.LOAD if load else MemOp.STORE, signed=signed,
            datasize=datasize, wback=wback, postindex=postindex,
        )


class LoadLiteralA64(LoadLiteral):
    """
    opc 0 1 1 V 0 0 imm19 Rt
    """

    @staticmethod
    def from_bitarray(instr, processor):
        opc = substring(instr, 31, 30)
        offset = sign_extend(substring(instr, 23, 5) << 2, 21, 64)
        t = substring(instr, 4, 0)
        if opc == 0b00:
            return LoadLiteralA64(instr, t=t, offset=offset, signed=False, datasize=32)
        if opc == 0b01:
            return LoadLiteralA64(instr, t=t, offset=offset, signed=False, datasize=64)
        if opc == 0b10:
            # LDRSW (literal)
            return LoadLiteralA64(instr, t=t, offset=offset, signed=True, datasize=32)
        # PRFM (literal)
        return LoadLiteralA64(instr, t=t, offset=offset, signed=False, datasize=64,
                              is_prefetch=True)


class LoadStoreExclusiveA64(LoadStoreExclusive):
    """
    size 0 0 1 0 0 0 o2 L o1 Rs o0 Rt2 Rn Rt

    Covers LDXR/STXR, their pair forms, and the plain acquire/release LDAR/STLR.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = substring(instr, 31, 30)
        o2 = bit_at(instr, 23)
        load = bit_at(instr, 22)
        pair = bit_at(instr, 21)

        # o2 set means an ordered access with no exclusive monitor involved.
        exclusive = not o2
        if o2 and pair:
            return None
        # A pair access is only defined for word and doubleword.
        if pair and size not in (0b10, 0b11):
            return None

        return LoadStoreExclusiveA64(
            instr, t=substring(instr, 4, 0), t2=substring(instr, 14, 10),
            n=substring(instr, 9, 5), s=substring(instr, 20, 16),
            memop=MemOp.LOAD if load else MemOp.STORE,
            datasize=8 << size, pair=bool(pair), exclusive=exclusive,
        )


# ----------------------------------------------------------------------------------
# SIMD and floating point variants (V = 1)
#
# These reuse the integer addressing modes but size differently: the access width comes
# from opc<1>:size, which reaches 128 bits for the Q registers rather than stopping at 64.
# ----------------------------------------------------------------------------------

from armulator.armv8.opcodes.abstract_opcodes.load_fp_literal import LoadFpLiteral  # noqa: E402
from armulator.armv8.opcodes.abstract_opcodes.load_store_fp_immediate import (  # noqa: E402
    LoadStoreFpImmediate,
)
from armulator.armv8.opcodes.abstract_opcodes.load_store_fp_pair import (  # noqa: E402
    LoadStoreFpPair,
)
from armulator.armv8.opcodes.abstract_opcodes.load_store_fp_register_offset import (  # noqa: E402
    LoadStoreFpRegisterOffset,
)


def decode_fp_size_opc(size, opc):
    """
    Resolve the SIMD/FP (size, opc) pair into (memop, datasize), or None when reserved.

    The scale is opc<1>:size, so a Q access is encoded as opc<1> set with size 00 -
    which is why a naive reading of size alone gets 128-bit accesses wrong.
    """
    scale = (bit_at(opc, 1) << 2) | size
    if scale > 4:
        return None
    memop = MemOp.LOAD if bit_at(opc, 0) else MemOp.STORE
    return memop, 8 << scale


class LoadStoreFpUnsignedOffsetA64(LoadStoreFpImmediate):
    """
    size 1 1 1 1 0 1 opc imm12 Rn Rt
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = substring(instr, 31, 30)
        decoded = decode_fp_size_opc(size, substring(instr, 23, 22))
        if decoded is None:
            return None
        memop, datasize = decoded
        scale = datasize.bit_length() - 4      # 8 -> 0, 16 -> 1, ... 128 -> 4
        return LoadStoreFpUnsignedOffsetA64(
            instr, t=substring(instr, 4, 0), n=substring(instr, 9, 5),
            offset=substring(instr, 21, 10) << scale, memop=memop,
            datasize=datasize, wback=False, postindex=False,
        )


class LoadStoreFpIndexedA64(LoadStoreFpImmediate):
    """
    size 1 1 1 1 0 0 opc 0 imm9 op4 Rn Rt - unscaled, post-indexed and pre-indexed.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        decoded = decode_fp_size_opc(substring(instr, 31, 30), substring(instr, 23, 22))
        if decoded is None:
            return None
        memop, datasize = decoded
        op4 = substring(instr, 11, 10)
        if op4 == 0b10:
            # There is no unprivileged form for SIMD and floating point registers.
            return None
        return LoadStoreFpIndexedA64(
            instr, t=substring(instr, 4, 0), n=substring(instr, 9, 5),
            offset=sign_extend(substring(instr, 20, 12), 9, 64), memop=memop,
            datasize=datasize, wback=op4 in (0b01, 0b11), postindex=op4 == 0b01,
        )


class LoadStoreFpRegisterOffsetA64(LoadStoreFpRegisterOffset):
    """
    size 1 1 1 1 0 0 opc 1 Rm option S 1 0 Rn Rt
    """

    @staticmethod
    def from_bitarray(instr, processor):
        size = substring(instr, 31, 30)
        decoded = decode_fp_size_opc(size, substring(instr, 23, 22))
        if decoded is None:
            return None
        memop, datasize = decoded
        option = substring(instr, 15, 13)
        if not bit_at(option, 1):
            return None
        scale = datasize.bit_length() - 4
        return LoadStoreFpRegisterOffsetA64(
            instr, t=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), extend_type=ExtendType(option),
            shift=scale if bit_at(instr, 12) else 0, memop=memop, datasize=datasize,
        )


class LoadStoreFpPairA64(LoadStoreFpPair):
    """
    opc 1 0 1 1 idx L imm7 Rt2 Rn Rt - opc gives 32, 64 or 128 bit elements.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        opc = substring(instr, 31, 30)
        if opc == 0b11:
            return None
        scale = 2 + opc                        # 00 -> S, 01 -> D, 10 -> Q
        datasize = 8 << scale
        idx = substring(instr, 24, 23)
        if idx == 0b01:
            wback, postindex = True, True
        elif idx == 0b11:
            wback, postindex = True, False
        else:
            wback, postindex = False, False
        return LoadStoreFpPairA64(
            instr, t=substring(instr, 4, 0), t2=substring(instr, 14, 10),
            n=substring(instr, 9, 5),
            offset=sign_extend(substring(instr, 21, 15), 7, 64) << scale,
            memop=MemOp.LOAD if bit_at(instr, 22) else MemOp.STORE,
            datasize=datasize, wback=wback, postindex=postindex,
        )


class LoadFpLiteralA64(LoadFpLiteral):
    """
    opc 0 1 1 1 0 0 imm19 Rt - opc 00 loads S, 01 loads D, 10 loads Q.
    """

    @staticmethod
    def from_bitarray(instr, processor):
        opc = substring(instr, 31, 30)
        if opc == 0b11:
            return None
        return LoadFpLiteralA64(
            instr, t=substring(instr, 4, 0),
            offset=sign_extend(substring(instr, 23, 5) << 2, 21, 64),
            datasize=8 << (2 + opc),
        )


from armulator.armv8.opcodes.abstract_opcodes.simd_load_store_multiple import (  # noqa: E402
    SimdLoadStoreMultiple,
)

#: opcode at instr[15:12] -> (rpt, selem).
#:
#: ``selem`` is the number of members per structure, which is the de-interleaving
#: stride; ``rpt`` is how many times the pattern repeats, used only by LD1 to move
#: several whole registers of contiguous data.
_STRUCTURE_SHAPE = {
    0b0000: (1, 4),   # LD4/ST4 - four-member structures
    0b0010: (4, 1),   # LD1/ST1 - four registers, contiguous
    0b0100: (1, 3),   # LD3/ST3
    0b0110: (3, 1),   # LD1/ST1 - three registers
    0b0111: (1, 1),   # LD1/ST1 - one register
    0b1000: (1, 2),   # LD2/ST2
    0b1010: (2, 1),   # LD1/ST1 - two registers
}


class SimdLoadStoreMultipleA64(SimdLoadStoreMultiple):
    """
    0 Q 0 0 1 1 0 0 (1) L (0) Rm opcode size Rn Rt

    Bit 23 selects the post-indexed form; in it Rm names a register holding the
    increment, or is 31 to mean "advance by the size of the transfer".
    """

    @staticmethod
    def from_bitarray(instr, processor):
        shape = _STRUCTURE_SHAPE.get(substring(instr, 15, 12))
        if shape is None:
            return None
        rpt, selem = shape
        post_index = bit_at(instr, 23)
        if not post_index and substring(instr, 20, 16) != 0:
            return None

        size = substring(instr, 11, 10)
        q = bit_at(instr, 30)
        element_size = 8 << size
        datasize = 128 if q else 64
        # The 1D arrangement (size 11 without Q) holds a single 64-bit element, which
        # only the single-structure forms can use.
        if size == 0b11 and not q and selem != 1:
            return None

        return SimdLoadStoreMultipleA64(
            instr, t=substring(instr, 4, 0), n=substring(instr, 9, 5),
            m=substring(instr, 20, 16), rpt=rpt, selem=selem,
            element_size=element_size, elements=datasize // element_size,
            memop=MemOp.LOAD if bit_at(instr, 22) else MemOp.STORE,
            datasize=datasize, wback=bool(post_index),
        )
