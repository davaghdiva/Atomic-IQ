from pathlib import Path
import numpy as np
from atomic_iq_sg import benchmarks


def test_rmt_uses_live_aspect_ratio():
    rng=np.random.default_rng(123)
    X=rng.normal(size=(24,8))
    S_auto,_=benchmarks.RMT1(X)
    S_explicit,_=benchmarks.RMT1(X,q=3.0)
    assert np.allclose(S_auto,S_explicit)


def test_no_numpy_matlib_dependency():
    root=Path(__file__).resolve().parents[1]
    txt=(root/'atomic_iq_sg'/'benchmarks.py').read_text()
    assert 'numpy.matlib' not in txt
    assert 'repmat' not in txt


def test_clean_asset_headers():
    root=Path(__file__).resolve().parents[1]
    import pandas as pd
    cols=list(pd.read_csv(root/'data'/'prices_multi_asset_master.csv',nrows=1).columns)
    assert cols==[c.strip() for c in cols]
