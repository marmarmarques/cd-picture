"""Scrambler and forward/inverse CIRC mapping against ECMA-130 and independent references."""

import numpy as np

from cdpicture import ecma130 as E
from cdpicture.writer import inverse_positions

M1, XA = E.SECTOR_LAYOUTS["mode1"], E.SECTOR_LAYOUTS["mode2form1"]


def test_annex_b_prefix_and_period():
    vector = bytes.fromhex("01 80 00 60 00 28 00 1e 80 08 60 06 a8 02 fe 81 80 60 60 28 28 1e 9e 88 68 66 ae aa fc 7f 01 e0")
    assert E.SCRAMBLE[12:44].tobytes() == vector
    assert E.lfsr_period() == 32767


def test_scrambler_matches_byte_at_a_time_formulation():
    # Cross-checked with redumper's cd_scrambler.ixx at fa5d2434c755a2dc26199fee2d20fc246a9e51c9.
    reg, reference = 1, []
    for _ in range(2340):
        reference.append(reg & 255)
        for _ in range(8):
            reg = (reg // 2) + (16384 if (reg & 1) != ((reg // 2) & 1) else 0)
    assert E.SCRAMBLE[12:].tobytes() == bytes(reference)


def test_clause_16_swap_before_circ():
    f, s = E.user_to_physical(0, np.array([0, 1]), M1)
    assert f.tolist() == [22, 19] and s.tolist() == [5, 4]


def test_delays_agree_with_figure_c4():
    ref = [-72, -143, -256, -327, -440, -511, -646, -717, -830, -901, -1014, -1085,
           -1556, -1627, -1740, -1811, -1924, -1995, -2130, -2201, -2314, -2385, -2498, -2569]
    assert (E.INPUT_SLOT - 24 * E.DELAY).tolist() == ref


def test_random_bytes_round_trip():
    rng = np.random.default_rng(713)
    lba = rng.integers(0, E.CAPACITY, 8000)
    byte = rng.integers(0, 2048, 8000)
    q = rng.integers(0, 256, 8000, dtype=np.uint8)
    mask = E.mask(M1)
    payload = q ^ mask[byte]
    frame, slot = E.user_to_physical(lba, byte, M1)
    back_lba, back_byte, ok = E.physical_to_user(frame, slot, M1)
    n = E.channel_position(frame, slot)
    assert ok.all() and np.array_equal(lba, back_lba) and np.array_equal(byte, back_byte)
    assert np.array_equal(q, payload ^ mask[back_byte])
    assert np.all(n % 588 == 51 + 17 * slot)


def test_mode2form1_layout():
    """User byte 0 of XA sits where Mode 1 byte 8 does; textures pre-scrambled at byte 24."""
    assert np.array_equal(E.mask(XA), E.SCRAMBLE[24:24 + 2048])
    assert E.user_to_physical(3, 0, XA) == E.user_to_physical(3, 8, M1)
    il, ib, _, _ = inverse_positions(97, 2, XA)
    f, s = E.user_to_physical(il, ib, XA)
    _, b, ok = E.physical_to_user(f, s, XA)
    assert len(il) == 2 * 2048 and ok.all() and np.array_equal(b, ib)


def test_inverse_enumeration_agrees_across_sector_boundaries():
    il, ib, ff, ss = inverse_positions(97, 5, M1)
    f, s = E.user_to_physical(il, ib, M1)
    assert len(il) == 5 * 2048 and np.array_equal(f, ff) and np.array_equal(s, ss)


def test_parity_slots_uncontrollable():
    _, _, ok = E.physical_to_user(np.arange(32), np.arange(32), M1)
    assert not np.any(ok[[12, 13, 14, 15, 28, 29, 30, 31]])


def test_last_raw_pair_and_read_tail():
    f, s = E.raw_to_physical(2351)
    assert int(f) == 202 and int(s) == 26 and E.read_end_bits(0) == 206 * 588
    f, s = E.raw_to_physical(2350)
    assert int(f) == 205 and int(s) == 27


def test_capacity_limits():
    for sectors in (20, E.CAPACITY + 1):
        try:
            E.check_capacity(sectors)
            assert False, sectors
        except ValueError:
            pass
