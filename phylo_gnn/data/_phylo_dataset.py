import os
from pathlib import Path
from typing import Any
from collections.abc import Callable
from numpy.typing import NDArray
import pandas as pd
from torch_geometric.data import InMemoryDataset, HeteroData  # type: ignore
import tqdm


DEFAULT_LABEL_NAMES = ["BD", "HE", "ME", "SAT", "SR", "WW"]


class CSVMetadata:
    def __init__(
        self,
        dataset_dir: Path | None = None,
        dataset_name: str = "",
        column_names: list[str] | None = None,
        label_names: list[str] | None = None,
        csv_filenames: list[str] | str | None = None,
        detect_csv_files_pattern: str = "*.csv",
        read_csv_kwargs: dict[str, Any] | None = None,
    ):
        self._dataset_dir = dataset_dir
        self._dataset_name = dataset_name
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
    def dataset_name(self) -> str:
        if not self._dataset_name:
            if len(self.csv_filenames) > 1:
                raise ValueError(
                    "Multiple CSV files detected. Please specify a dataset "
                    "name."
                )
            return self.csv_filenames[0].split(".")[0]
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value: str) -> None:
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


class PhyloCSVDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        process_function: Callable[[str, NDArray], HeteroData],
        csv_metadata: CSVMetadata,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,
    ):
        if not callable(process_function):
            raise ValueError("`process_function` must be a callable function.")

        # Check that
        self.csv_metadata = csv_metadata
        self.process_func = process_function
        self._processed_paths_abs_cache: list[str] = []
        self.df = self.csv_metadata.get_combined_dataframe()
        super().__init__(
            root, transform, pre_transform, pre_filter, log, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> list[str]:
        return [
            os.path.join(self.csv_metadata.dataset_dir, f)
            for f in self.csv_metadata.csv_filenames
        ]

    @property
    def processed_file_names(self) -> list[str]:
        return [self.csv_metadata.dataset_name + ".pt"]

    def download(self):
        pass

    def process(self):
        data_list = []
        for _, row in tqdm.tqdm(
            self.df.iterrows(),
            desc="Processing newicks...",
            disable=not self.log,
            total=len(self.df),
        ):
            newick = row[self.csv_metadata.column_names[0]]
            # Other columns are the target
            label = row[self.csv_metadata.column_names[1:]]

            data = self.process_func(newick, label.to_numpy())
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
