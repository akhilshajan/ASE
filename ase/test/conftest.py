import os
import pytest
from ase.utils import workdir
from pathlib import Path
from ase.test.factories import (Factories, CalculatorInputs,
                                make_factory_fixture, get_testing_executables)


@pytest.fixture(scope='session')
def enabled_calculators():
    from ase.calculators.calculator import names as calculator_names
    all_names = set(calculator_names)
    names = set(os.environ.get('ASE_TEST_CALCULATORS', '').split())
    for name in names:
        assert name in all_names
    return names


@pytest.fixture(scope='session', autouse=True)
def monkeypatch_disabled_calculators(request, enabled_calculators):
    from ase.test.testsuite import disable_calculators, test_calculator_names
    from ase.calculators.calculator import names as calculator_names
    test_calculator_names += enabled_calculators
    disable_calculators([name for name in calculator_names
                         if name not in enabled_calculators])


# Backport of tmp_path fixture from pytest 3.9.
# We want to be compatible with pytest 3.3.2 and pytest-xdist 1.22.1.
# These are provided with Ubuntu 18.04.
# Current Debian stable uses a newer libraries, so that should be OK.
@pytest.fixture
def tmp_path(tmpdir):
    # Avoid trouble since tmpdir can be a py._path.local.LocalPath
    return Path(str(tmpdir))


@pytest.fixture(autouse=True)
def use_tmp_workdir(tmp_path):
    # Pytest can on some systems provide a Path from pathlib2.  Normalize:
    path = Path(str(tmp_path))
    with workdir(path, mkdir=True):
        yield tmp_path


@pytest.fixture(scope='session')
def tkinter():
    import tkinter
    try:
        tkinter.Tk()
    except tkinter.TclError as err:
        pytest.skip('no tkinter: {}'.format(err))


@pytest.fixture(scope='session')
def plt(tkinter):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg', warn=False)

    import matplotlib.pyplot as plt
    return plt


@pytest.fixture
def figure(plt):
    fig = plt.figure()
    yield fig
    plt.close(fig)


@pytest.fixture(scope='session')
def psycopg2():
    return pytest.importorskip('psycopg2')


@pytest.fixture(scope='session')
def datafiles():
    try:
        import asetest
    except ImportError:
        return {}
    else:
        return asetest.datafiles.paths


@pytest.fixture(scope='session')
def configured_executables():
    return get_testing_executables()


@pytest.fixture(scope='session')
def factories(configured_executables, datafiles, enabled_calculators):
    return Factories(configured_executables, datafiles)


abinit_factory = make_factory_fixture('abinit')
cp2k_factory = make_factory_fixture('cp2k')
espresso_factory = make_factory_fixture('espresso')
gpaw_factory = make_factory_fixture('gpaw')
octopus_factory = make_factory_fixture('octopus')
siesta_factory = make_factory_fixture('siesta')


@pytest.fixture
def code(request, factories):
    name, kwargs = request.param
    factory = factories[name]
    return CalculatorInputs(factory, kwargs)


def pytest_generate_tests(metafunc):
    from ase.test.factories import parametrize_calculator_tests
    parametrize_calculator_tests(metafunc)
