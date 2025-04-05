import os
from collections.abc import Callable
from numpy.typing import NDArray
from torch_geometric.data import InMemoryDataset, HeteroData  # type: ignore
import tqdm

from phylo_gnn.data import CSVMetadata


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
