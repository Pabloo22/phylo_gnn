import os
from pathlib import Path
from typing import Any
import pandas as pd


DEFAULT_LABEL_NAMES = ["BD", "HE", "ME", "SAT", "SR", "WW"]


class CSVMetadata:

    def __init__(
        self,
        dataset_dir: Path | None = None,
        processed_filename: str = "",
        column_names: list[str] | None = None,
        label_names: list[str] | None = None,
        csv_filenames: list[str] | str | None = None,
        detect_csv_files_pattern: str = "*.csv",
        read_csv_kwargs: dict[str, Any] | None = None,
    ):
        self._dataset_dir = dataset_dir
        self._dataset_name = processed_filename
        self.column_names = (
            column_names if column_names is not None else ["nwk", "label"]
        )
        self.label_names = (
            label_names if label_names is not None else DEFAULT_LABEL_NAMES
        )
        if isinstance(csv_filenames, str):
            csv_filenames = [csv_filenames]
        self.csv_filenames = csv_filenames if csv_filenames is not None else []
        self.detect_csv_files_pattern = detect_csv_files_pattern
        self.read_csv_kwargs = (
            read_csv_kwargs if read_csv_kwargs is not None else {}
        )

        if not self.csv_filenames:
            self.csv_filenames = self._detect_csv_files()

    @property
    def dataset_dir(self) -> Path:
        if self._dataset_dir is None:
            raise ValueError("`dataset_dir` must be set.")
        return self._dataset_dir

    @dataset_dir.setter
    def dataset_dir(self, value: str | Path) -> None:
        if isinstance(value, str):
            value = Path(value)
        self._dataset_dir = value

    @property
    def processed_file_name(self) -> str:
        if not self._dataset_name:
            if len(self.csv_filenames) > 1:
                raise ValueError(
                    "Multiple CSV files detected. Please specify a dataset "
                    "name."
                )
            return self.csv_filenames[0].split(".")[0]
        return self._dataset_name

    @processed_file_name.setter
    def processed_file_name(self, value: str) -> None:
        self._dataset_name = value

    def _detect_csv_files(self) -> list[str]:
        return [
            str(f)
            for f in self.dataset_dir.glob(self.detect_csv_files_pattern)
        ]

    def get_combined_dataframe(self) -> pd.DataFrame:
        dfs = []
        for csv_file in self.csv_filenames:
            csv_path = os.path.join(self.dataset_dir, csv_file)
            df = pd.read_csv(csv_path, **self.read_csv_kwargs)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
