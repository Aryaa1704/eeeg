import numpy as np
from eeg_mi.model import crop, filter_trials
from eeg_mi.stats import mcnemar_exact, holm
def test_crop_and_filter_shape():
    x=np.random.default_rng(0).normal(size=(4,3,1750)); assert crop(x,250,(3,6)).shape == (4,3,750)
    assert filter_trials(x,250,(8,30),3).shape == x.shape
def test_stats():
    r=mcnemar_exact(np.array([True,False]),np.array([False,True])); assert r['p_value']==1
    assert holm([.01,.04]) == [.02,.04]
