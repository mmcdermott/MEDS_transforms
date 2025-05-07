"""Test set-up and fixtures code."""

import json
import tempfile
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from omegaconf import DictConfig


@contextmanager
def print_warnings(caplog: pytest.LogCaptureFixture):
    """Captures all logged warnings within this context block and prints them upon exit.

    This is useful in doctests, where you want to show printed outputs for documentation and testing purposes.
    """

    n_current_records = len(caplog.records)

    with caplog.at_level("WARNING"):
        yield
    # Print all captured warnings upon exit
    for record in caplog.records[n_current_records:]:
        print(f"Warning: {record.getMessage()}")


@pytest.fixture(autouse=True)
def __MEDS_transforms_setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    simple_static_MEDS,
) -> None:
    doctest_namespace.update(
        {
            "DictConfig": DictConfig,
            "MagicMock": MagicMock,
            "patch": patch,
            "simple_static_MEDS": simple_static_MEDS,
            "print_warnings": partial(print_warnings, caplog),
            "json": json,
            "pl": pl,
            "datetime": datetime,
            "tempfile": tempfile,
        }
    )
