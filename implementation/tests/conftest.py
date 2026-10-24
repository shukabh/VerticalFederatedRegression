import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..')))
sys.path.insert(0, os.path.join(HERE, 'reference'))


@pytest.fixture(scope='session')
def workdir(tmp_path_factory):
    return str(tmp_path_factory.mktemp('vfr'))


@pytest.fixture
def debug_env(monkeypatch):
    """Enables the guarded test-only paths (eta = 0, c0' fingerprints).  Never set in production."""
    monkeypatch.setenv('VFR_DEBUG_NO_DP', '1')
    yield
