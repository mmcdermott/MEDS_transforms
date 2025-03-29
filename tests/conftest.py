from MEDS_transforms.__main__ import get_all_stages

REGISTERED_STAGES = get_all_stages()


def pytest_addoption(parser):
    stage_help_str = (
        "Specify the MEDS-transform stage(s) to run when testing registered stages. If not specified, all "
        "stages will be run. This option can be used multiple times to specify multiple stages; e.g., "
        "'--test_stage stage1 --test_stage stage2'. The stages are specified by their registered names. "
        "This option only affects the automated testing of all registered stages, and does not impact other "
        "tests, such as the runner or any manual tests. Stages specified here but that lack static data "
        "examples will be skipped."
    )
    parser.addoption("--test_stage", action="append", type=str, default=[], help=stage_help_str)


def pytest_generate_tests(metafunc):
    """Generate tests for registered stages based on the command line options."""
    all_stages = sorted(list(REGISTERED_STAGES.keys()))

    if "stage" in metafunc.fixturenames:
        # Get the stages to test from the command line options
        test_stages = metafunc.config.getoption("test_stage")
        if test_stages:
            if not all(stage in all_stages for stage in test_stages):
                raise ValueError(
                    f"Invalid stage(s) specified: {', '.join(test_stages)}. "
                    f"Available stages are: {', '.join(all_stages)}."
                )
            # Filter the registered stages based on the command line options
            metafunc.parametrize("stage", test_stages)
        else:
            # If no specific stages are provided, run all registered stages
            metafunc.parametrize("stage", all_stages)
