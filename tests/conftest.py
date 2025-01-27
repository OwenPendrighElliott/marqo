import pytest

def pytest_addoption(parser):
    parser.addoption("--largemodel", action="store_true", default = False)


def pytest_configure(config):
    config.addinivalue_line("markers", "largemodel: mark test as largemodels")


def pytest_collection_modifyitems(config, items):
    skip_largemodel = pytest.mark.skip(reason="need --largemodel option to run")
    skip_cpu_only = pytest.mark.skip(reason="skip in --largemodel mode when cpu_only is present")

    if config.getoption("--largemodel"):
        # --largemodel given in cli: do not skip largemodel tests, skip cpu_only tests
        for item in items:
            if "cpu_only" in item.keywords:
                item.add_marker(skip_cpu_only)
    else:
        for item in items:
            if "largemodel" in item.keywords:
                item.add_marker(skip_largemodel)