"""Test set-up and fixtures code."""

import json
import tempfile
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from typing import Any

import polars as pl
import pytest


@contextmanager
def print_warnings(caplog: pytest.LogCaptureFixture):
    """Captures all logged warnings within this context block and prints them upon exit.

    This is useful in doctests, where you want to show printed outputs for documentation and testing purposes.
    """

    N_current_records = len(caplog.records)

    with caplog.at_level("WARNING"):
        yield
    # Print all captured warnings upon exit
    for record in caplog.records[N_current_records:]:
        print(f"Warning: {record.getMessage()}")


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(doctest_namespace: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    doctest_namespace.update(
        {
            "print_warnings": partial(print_warnings, caplog),
            "json": json,
            "pl": pl,
            "datetime": datetime,
            "tempfile": tempfile,
        }
    )
