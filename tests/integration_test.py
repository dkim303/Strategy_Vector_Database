import pytest
import numpy as np
import pandas as pd
import psycopg

from vector_db_package.database_utils import (
    get_config,
)

def initial_test():
    assert 1 == 1

def test_config():
    config = get_config("first_config.yml")

    assert config is not None