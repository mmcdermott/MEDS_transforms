import importlib.metadata

from .. import __package_name__


class StageNotFoundError(Exception):
    """Custom error for when a stage is not found."""


class StageDiscoveryError(Exception):
    """Custom error for stage discovery."""


def get_all_registered_stages() -> dict[str, importlib.metadata.EntryPoint]:
    """Returns a dictionary mapping stage name to entry point for all registered MEDS-transforms stages.

    Registration here _does not mean simply having used the `Stage.register` decorator_. Rather, it means
    having added a `MEDS_transform.stages` entry point to your python package. E.g., in your `pyproject.toml`,
    you could have:

    ```toml
    [project.entry-points."MEDS_transforms.stages"]
    aggregate_code_metadata = "MEDS_transforms.stages:aggregate_code_metadata"
    fit_vocabulary_indices = "MEDS_transforms.stages:fit_vocabulary_indices"
    ```

    If so, then this function will include the stage names `aggregate_code_metadata` and
    `fit_vocabulary_indices` in the returned dictionary, mapping to the entry point objects that you specify.

    For the examples below, we will merely mock the entry points so we can showcase behaviors, but in practice
    this would be read dynamically across all installed packages in the python context.

    Examples:

        First, we set up some fake entry-points, to show that the right entry points are selected and
        returned:

        >>> fake_eps = {
        ...     "wrong_package.stages": [("A", "stage_a"), ("B", "stage_b")], # [(name, entry_point), ...]
        ...     "MEDS_transforms.stages": [("C", "stage_c"), ("D", "stage_d")],
        ... }
        >>> def get_group(group: str):
        ...     vals = fake_eps[group]
        ...     out = MagicMock()
        ...     out.names = [name for name, _ in vals]
        ...     out.select = lambda name: [entry_point for n, entry_point in vals if n == name]
        ...     out.__getitem__ = lambda self, name: next(ep for n, ep in vals if n == name)
        ...     return out
        >>> with patch("importlib.metadata.entry_points") as mock_entry_points:
        ...     mock_entry_points.side_effect = get_group
        ...     get_all_registered_stages()
        {'C': 'stage_c', 'D': 'stage_d'}

        If there are duplicate stage names, we get an error:

        >>> fake_eps["MEDS_transforms.stages"].append(("C", "stage_c_2"))
        >>> with patch("importlib.metadata.entry_points") as mock_entry_points:
        ...     mock_entry_points.side_effect = get_group
        ...     get_all_registered_stages()
        Traceback (most recent call last):
            ...
        MEDS_transforms.stages.discovery.StageDiscoveryError: Multiple entry points registered for the same
        stage:
          - C: ['stage_c', 'stage_c_2']
    """
    eps = importlib.metadata.entry_points(group=f"{__package_name__}.stages")

    errors = {}
    for name in eps.names:
        if len(eps.select(name=name)) > 1 and name not in errors:
            errors[name] = eps.select(name=name)
    if errors:
        error_str = "\n".join(f"  - {name}: {eps}" for name, eps in errors.items())
        raise StageDiscoveryError(f"Multiple entry points registered for the same stage:\n{error_str}")

    return {n: eps[n] for n in eps.names}
