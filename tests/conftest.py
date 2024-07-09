# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest
import numpy as np

def pytest_addoption(parser):
    parser.addoption('--rng-entropy',
                     help='Entropy for random seed generator.')

@pytest.fixture
def np_rng(request):
    """Instance of numpy.random.Generator."""
    e = request.config.getoption('--rng-entropy')
    print('\nrng entropy: ',sq := np.random.SeedSequence(e if e is None else int(e)).entropy)
    return np.random.default_rng(seed=sq)
