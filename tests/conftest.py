# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest
import numpy as np
import damask
from damask import tensor

from skel import _conventions
from skel import Mandel

def pytest_addoption(parser):
    parser.addoption('--rng-entropy',
                     help='Entropy for random seed generator.')

@pytest.fixture
def np_rng(request):
    """Instance of numpy.random.Generator."""
    e = request.config.getoption('--rng-entropy')
    print('\nrng entropy: ',sq := np.random.SeedSequence(e if e is None else int(e)).entropy)
    return np.random.default_rng(seed=sq)

@pytest.fixture
def random_shape(np_rng):
    """Factory for random batch shapes."""
    def _random_shape(ndim):
        return tuple(np_rng.integers(4,11,size=ndim))
    return _random_shape

@pytest.fixture
def random_6(np_rng):
    """Factory for random vectors of length 6."""
    def _random_6(shape=()):
        return np_rng.random(shape+(6,))
    return _random_6

@pytest.fixture
def symmetric_2nd(np_rng):
    """Factory for random symmetric 2nd order tensors."""
    def _symmetric_2nd(shape=()):
        return tensor.symmetric(np_rng.random(shape+(3,3)))
    return _symmetric_2nd

@pytest.fixture
def symmetric_6x6(np_rng):
    """Factory for random symmetric 6x6 matrices."""
    def _symmetric_6x6(shape=()):
        return tensor.symmetric(np_rng.random(shape+(6,6)))
    return _symmetric_6x6

@pytest.fixture
def symmetric_4th(np_rng):
    """Factory for random symmetric 4th order tensors."""
    def _symmetric_4th(shape=()):
        return _conventions.convert_6x6_to_3x3x3x3(tensor.symmetric(np_rng.random(shape+(6,6))),Mandel.BASIS)
    return _symmetric_4th

@pytest.fixture
def random_rotation(np_rng):
    """Factory for random orientations."""
    def _random_rotation(shape=()):
        return damask.Rotation.from_random(shape,np_rng)
    return _random_rotation

@pytest.fixture
def assert_allclose():
    """
    Asserts the element-wise equality of two arrays within relative+absolute tolerance.

    Parameters
    ----------
    a : np.ndarray or h5py Dataset
        Array to compare.
    b : np.ndarray or h5py Dataset
        Array to compare.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    equal_nan : bool
        Whether NaNs in the same positions compare as equal.
        Defaults to False.
    N : int
        Maximum number of reported deviating values.
        Defaults to 16. N=None outputs all values.
    msg : str
        Informative message.

    """
    def _assert_allclose(a,b,
                         rtol: float = 1e-05,
                         atol: float = 1e-08,
                         equal_nan: bool = False,
                         N = 16,
                         msg = '',
                         ):
        assert np.logical_or(np.isclose(a,b, rtol=rtol,atol=atol,equal_nan=equal_nan),
                             np.isclose(b,a, rtol=rtol,atol=atol,equal_nan=equal_nan)).all(), \
               report_nonclose(a,b, rtol=rtol,atol=atol, equal_nan=equal_nan, N=N, msg=msg)

    return _assert_allclose

def report_nonclose(a, b, rtol, atol, equal_nan, N, msg):
    """
    Report where values of two arrays deviate from each other.

    Output is sorted from large to small magnitude of absolute difference.

    Parameters
    ----------
    a : np.ndarray or h5py Dataset
        Array to compare.
    b : np.ndarray or h5py Dataset
        Array to compare.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    equal_nan : bool
        Whether NaNs in the same positions compare as equal.
    N : int
        Maximum number of reported deviating values.
        Defaults to all.
    msg : str
        Informative message added to output.
        Defaults to None.

    Returns
    -------
    description : str
        Description of differences.

    """
    a_,b_ = np.broadcast_arrays(np.atleast_1d(np.asarray(a)),
                                np.atleast_1d(np.asarray(b)))

    absdiff = np.abs(a_ - b_)
    absmax  = np.max(np.abs(np.stack((a_, b_))), axis=0)

    idx = np.logical_not(
              np.logical_or(np.isclose(a_,b_,rtol=rtol,atol=atol,equal_nan=equal_nan),
                            np.isclose(b_,a_,rtol=rtol,atol=atol,equal_nan=equal_nan))
            ).nonzero()
    ids = np.argsort(absdiff[idx])[::-1]
    n = len(idx[0])
    split = N is not None and N<n
    head,tail = (slice(0,(N+1)//2),slice(n-N//2,n)) if split else (slice(0,n),slice(n,n))
    diffs = [ f'abs / rel diff {absdiff[idx][i]:>16.10g} / {absdiff[idx][i]/absmax[idx][i]:<16.10g}'
             +f' between {a_[idx][i]:>16.10g} and {b_[idx][i]:<16.10g}'
             +f' at {np.transpose(idx)[i]}' for i in list(ids[head])+list(ids[tail])]
    if split: diffs.insert(head.stop,'...')

    return '\n'.join( ['',f'fraction {n}/{a_.size} outside tolerance']
                     +[f'tolerance: atol = {atol} + rtol = {rtol} * absmax']
                     +['']+diffs
                     +(['',msg] if msg is not None else [])
                     )

