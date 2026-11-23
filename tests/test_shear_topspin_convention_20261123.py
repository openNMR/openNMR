"""MQMAS shear: the F1 direction TopSpin's own ``xfshear`` uses.

Found on a real 27Al 3QMAS (YAG, States-TPPI, I = 5/2): with the default
``f1_sign = +1`` openNMR smeared the sites into diagonal streaks, while
``f1_sign = -1`` reproduced TopSpin's ``xfshear`` result (correlation 0.94
against 0.22, peaks within ~0.3 ppm, best kappa exactly 19/12). Bruker
MQMAS/STMAS data therefore start at -1.

The reference here is a numpy port of xfshear's arithmetic (the AU
program multiplies the F2-transformed, still-t1 data by
exp(i (ph1 k + phc2) 2n), ph1 = -pi R SW_p / SWH1 / SI2, k counted from
the centre of the stored F2 axis), written independently of
``shear_2d``'s Fourier-shift route.
"""

import numpy as np
import pytest

from opennmr.notebook import shearing as sh

from test_shearing import make_2d, two_site_mqmas

R_3Q_52 = 19 / 12


def _xfshear_reference(crystallites, *, n1, n2, zf1, zf2, sw1, sw2, ratio, t1_decay=2.0e-3,
                       t2_decay=1.5e-3):
    """F2 spectrum x t1 data, phase-rotated like xfshear, then F1-FT'd."""
    N1, N2 = n1 * zf1, n2 * zf2
    t1 = np.arange(n1) / sw1
    t2 = np.arange(n2) / sw2
    S = np.zeros((N1, N2), complex)
    for nu1, nu2, w in crystallites:
        f2 = np.zeros(N2, complex)
        f2[:n2] = np.exp(-1j * 2 * np.pi * nu2 * t2) * np.exp(-t2 / t2_decay)
        l2 = np.real(np.fft.fftshift(np.fft.fft(f2)))
        keep = np.abs(l2) > 1e-6 * np.abs(l2).max()
        col = np.zeros((N1, N2), complex)
        col[:n1, keep] = np.exp(-1j * 2 * np.pi * nu1 * t1)[:, None] * np.exp(-t1 / t1_decay)[:, None]
        S += w * col * l2[None, :]
    k = np.arange(N2) - N2 // 2                       # centred, stored (descending) order
    ph1 = -np.pi * ratio * sw2 / sw1 / N2             # xfshear: -pi R SW_p / SWH1 / SI2
    S = S * np.exp(1j * ph1 * k[None, :] * 2 * np.arange(N1)[:, None])
    return np.fft.fftshift(np.fft.fft(S, axis=0), axes=0)


def test_default_f1_sign_follows_topspin_for_bruker_mqmas():
    spec = make_2d([(1000.0, 500.0, 1.0)])
    assert spec.metadata.vendor == "bruker"
    assert sh.default_f1_sign(spec, "mqmas") == -1
    assert sh.default_f1_sign(spec, "stmas") == -1
    assert sh.default_f1_sign(spec, "jres") == 1          # unverified experiments keep +1
    assert sh.default_f1_sign(None, "mqmas") == 1
    import dataclasses
    other = dataclasses.replace(spec, metadata=dataclasses.replace(spec.metadata, vendor="jeol"))
    assert sh.default_f1_sign(other, "mqmas") == 1


def test_plan_takes_the_default_from_the_spectrum_and_explicit_wins():
    spec = make_2d([(1000.0, 500.0, 1.0)])
    plan = sh.make_shear_plan("mqmas", spin=2.5, mq_order=3, spectrum=spec)
    assert plan.f1_sign == -1 and plan.kappa_t2 == pytest.approx(+R_3Q_52)
    assert plan.scale_f1 == pytest.approx(17 / 12)         # not reversed, as in TopSpin
    plain = sh.make_shear_plan("mqmas", spin=2.5, mq_order=3)
    assert plain.f1_sign == 1 and plain.kappa_t2 == pytest.approx(-R_3Q_52)
    forced = sh.make_shear_plan("mqmas", spin=2.5, mq_order=3, spectrum=spec, f1_sign=1)
    assert forced.kappa_t2 == pytest.approx(-R_3Q_52)


def test_shear_matches_a_numpy_port_of_xfshear():
    n1, n2, zf1, zf2, sw1, sw2 = 48, 200, 2, 2, 20000.0, 50000.0
    cryst = [(1500.0, 2000.0, 1.0), (-2500.0, -3000.0, 0.6), (300.0, 700.0, 0.8)]
    spec = make_2d(cryst, n1=n1, n2=n2, zf1=zf1, zf2=zf2, sw1=sw1, sw2=sw2)
    plan = sh.make_shear_plan("mqmas", spin=2.5, mq_order=3, spectrum=spec)
    out = sh.shear_2d(spec, plan)
    ref = _xfshear_reference(cryst, n1=n1, n2=n2, zf1=zf1, zf2=zf2, sw1=sw1, sw2=sw2,
                             ratio=R_3Q_52)
    got = out.data
    assert got.shape == ref.real.shape
    assert out.axes[0].ppm[0] > out.axes[0].ppm[-1]        # still descending (no reversal)
    scale = np.abs(ref.real).max()
    assert np.max(np.abs(got - ref.real)) / scale < 2e-3


def test_the_other_sign_does_not_reproduce_xfshear():
    n1, n2, zf1, zf2, sw1, sw2 = 48, 200, 2, 2, 20000.0, 50000.0
    cryst = [(1500.0, 2000.0, 1.0), (-2500.0, -3000.0, 0.6)]
    spec = make_2d(cryst, n1=n1, n2=n2, zf1=zf1, zf2=zf2, sw1=sw1, sw2=sw2)
    wrong = sh.shear_2d(spec, sh.make_shear_plan("mqmas", spin=2.5, mq_order=3, f1_sign=1))
    ref = _xfshear_reference(cryst, n1=n1, n2=n2, zf1=zf1, zf2=zf2, sw1=sw1, sw2=sw2,
                             ratio=R_3Q_52).real
    # the reversed plan flips F1; compare both ways round and demand a clear mismatch
    err = min(np.abs(wrong.data - ref).max(), np.abs(wrong.data[::-1] - ref).max())
    assert err / np.abs(ref).max() > 0.2
