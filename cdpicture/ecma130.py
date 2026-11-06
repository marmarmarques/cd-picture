"""CD-ROM user bytes -> physical CIRC data symbols (ECMA-130, 2nd ed.).

From disc-probe's cd.py, implemented there from clauses 14-19 and Annexes B/C, especially
figure C.4: https://ecma-international.org/wp-content/uploads/ECMA-130_2nd_edition_june_1996.pdf

Clause 16 swaps each adjacent pair AFTER scrambling, before CIRC. Sector byte 0 sits at
unswapped frame position 0 (F1 position 1). Positions use the forward encoder's 3..108-frame
delays. Headers, EDC, sector ECC, CIRC parity, subcode and sync are not controllable payload;
this module maps data symbols, it does not simulate parity or EFM/DSV.

Unlike disc-probe, the sector layout is not module state: every function that depends on
where the 2048 user bytes sit in the 2352-byte sector takes that `offset` explicitly.
"""

from __future__ import annotations

import numpy as np

SECTOR_BYTES = 2048
RAW_BYTES = 2352
FRAME_BITS = 588
FRAMES_PER_SECTOR = 98
SECTOR_BITS = FRAME_BITS * FRAMES_PER_SECTOR
CAPACITY = 359849  # ATIP 79:59:74 (80 min CD-R); no overburn
BIT_NM = 1200e6 / 4321800  # 1.2 m/s, 4.3218 MHz
R0_MM = 25.0
PITCH_UM = 1.6
READ_TAIL_FRAMES = 108
FOOTPRINT_FRAMES = np.arange(3, 206, 8)  # frames past LBA*98 that hold a sector's bytes

# Where the user bytes start in the raw sector. Mode 1 (ImgBurn, cdrecord) has them at 16;
# Mode 2 Form 1 / XA (Windows IMAPI2 data burns) at 24, after an 8-byte subheader. Textures
# pre-scrambled for the wrong layout turn into noise, so match the burner.
SECTOR_LAYOUTS = {"mode1": 16, "mode2form1": 24}


def _step(reg):
    return (reg >> 1) | (((reg ^ (reg >> 1)) & 1) << 14)


def scramble_sequence(count=2340):
    """Annex B: seed 1 after EACH sector sync, low bit first in every byte."""
    out = np.zeros(count, np.uint8)
    reg = 1
    for i in range(count):
        for bit in range(8):
            out[i] |= (reg & 1) << bit
            reg = _step(reg)
    return out


SCRAMBLE = np.concatenate([np.zeros(12, np.uint8), scramble_sequence()])


def lfsr_period():
    reg, count = 1, 0
    while True:
        reg, count = _step(reg), count + 1
        if reg == 1:
            return count


def mask(offset):
    """Scrambler bytes over the 2048 user bytes: recorded = payload ^ mask."""
    return SCRAMBLE[offset:offset + SECTOR_BYTES]


# Figure C.4: output data slots name words 0,4,8,1,5,9,2,6,10,3,7,11.
# Q occupies F2 slots 12..15; P occupies 28..31. Every even F2 slot has
# the final one-frame delay; words 0,4,8,1,5,9 have the initial two frames.
DATA_SLOTS = np.r_[0:12, 16:28]
WORD_ORDER = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
INPUT_SLOT = (2 * WORD_ORDER[:, None] + np.arange(2)).ravel()
DELAY = 4 * DATA_SLOTS + (DATA_SLOTS < 12) * 2 + (DATA_SLOTS % 2 == 0)
_OUT_FOR_INPUT = np.empty(24, np.int64)
_DELAY_FOR_INPUT = np.empty(24, np.int64)
_OUT_FOR_INPUT[INPUT_SLOT] = DATA_SLOTS
_DELAY_FOR_INPUT[INPUT_SLOT] = DELAY
_IN_FOR_OUT = np.full(32, -1, np.int64)
_IN_FOR_OUT[DATA_SLOTS] = INPUT_SLOT
_DELAY_FOR_OUT = np.zeros(32, np.int64)
_DELAY_FOR_OUT[DATA_SLOTS] = DELAY


def raw_to_physical(raw_index):
    """Absolute raw-sector byte index -> (channel frame, F2 symbol slot)."""
    swapped = np.asarray(raw_index, dtype=np.int64) ^ 1
    frame, slot = swapped // 24, swapped % 24
    return frame + _DELAY_FOR_INPUT[slot], _OUT_FOR_INPUT[slot]


def physical_to_raw(frame, slot):
    """Inverse data path. Parity slots return -1; negative lead-in indices are valid."""
    frame, slot = np.broadcast_arrays(np.asarray(frame, np.int64), np.asarray(slot, np.int64))
    if np.any((slot < 0) | (slot >= 32)):
        raise ValueError("F2 slot must be in 0..31")
    inp = _IN_FOR_OUT[slot]
    raw = ((frame - _DELAY_FOR_OUT[slot]) * 24 + inp) ^ 1
    return np.where(inp >= 0, raw, -1)


def user_to_physical(lba, byte, offset):
    """User byte -> (frame, F2 slot); lba and byte broadcast."""
    byte = np.asarray(byte, np.int64)
    if np.any((byte < 0) | (byte >= SECTOR_BYTES)):
        raise ValueError("user byte must be in 0..2047")
    return raw_to_physical(np.asarray(lba, np.int64) * RAW_BYTES + offset + byte)


def physical_to_user(frame, slot, offset):
    """(lba, user offset, controllable); headers/ECC/parity/lead-in are false."""
    raw = physical_to_raw(frame, slot)
    lba, byte = raw // RAW_BYTES, raw % RAW_BYTES - offset
    ok = (raw >= 0) & (byte >= 0) & (byte < SECTOR_BYTES)
    return lba, byte, ok


def channel_position(frame, slot, centre=True):
    """EFM word start (+7 for centre), after 24 sync + 3 merge + 17 subcode."""
    return np.asarray(frame) * FRAME_BITS + 44 + 17 * np.asarray(slot) + (7 if centre else 0)


def read_end_bits(lba, count=1):
    """Model assumption: full raw sector decode ends after its last CIRC frame.

    The last raw byte pair reaches F2 slot 27 at input frame 97+108; the end of that frame
    is (LBA+1)*98+108. Actual firmware completion/sector origin is unmeasured.
    """
    return ((np.asarray(lba) + count) * FRAMES_PER_SECTOR + READ_TAIL_FRAMES) * FRAME_BITS


def check_capacity(sectors, capacity=CAPACITY):
    if not 21 <= sectors <= min(capacity, CAPACITY):
        raise ValueError(f"CD image needs 21..{min(capacity, CAPACITY)} sectors, got {sectors}")
