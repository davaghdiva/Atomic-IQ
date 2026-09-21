"""Small semantic regression checks referenced in the Atomic-IQ manuscript.

Run from the root of the frozen Atomic-IQ computational repository so that
``atomic_iq_sg`` is importable.  These checks alter no estimator or empirical
result; they verify exact dependence semantics of the published construction.
"""
import numpy as np
from atomic_iq_sg.estimator import correlation

PARAMS = dict(c=0.40, delta=1.25, p=0.10, chi=0.90,
              state_scale='mad', target_min_eig=0.0)


def test_identical_and_opposite():
    x = np.linspace(-2.0, 2.0, 20)
    X = np.column_stack([x, x, -x])
    C = correlation(X, **PARAMS)
    assert abs(C[0, 1] - 1.0) < 1e-12
    assert abs(C[0, 2] + 1.0) < 1e-12


def test_universe_invariance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    C1 = correlation(X, **PARAMS)
    X2 = np.column_stack([X, rng.normal(0.0, 50.0, size=(20, 3))])
    C2 = correlation(X2, **PARAMS)
    assert np.allclose(C1, C2[:2, :2], atol=1e-12, rtol=1e-12)


def test_duplicate_asset():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 4))
    C1 = correlation(X, **PARAMS)
    X2 = np.column_stack([X, X[:, 1]])
    C2 = correlation(X2, **PARAMS)
    assert np.allclose(C1, C2[:4, :4], atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    test_identical_and_opposite()
    test_universe_invariance()
    test_duplicate_asset()
    print('All Atomic-IQ semantic checks passed.')
