

import pandas as pd
from jutils.data import DataUtils
from jutils.visual import Plot
from sklearn.model_selection import train_test_split
from pandas_profiling.profile_report import ProfileReport
from pathlib import Path
import numpy as np

du = DataUtils(
    Path(r'..\data').resolve().absolute(),
    "kc_house_dataDS.parquet",
    lambda path: pd.read_parquet(path),
    lambda df, path: df.to_parquet(path)
)
du.data = du.load_data(du.interim_path.joinpath(du.input_file_name))

du.data