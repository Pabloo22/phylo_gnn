import pathlib
from typing import List, Optional


class ClassificationFolderMetadata:
    """Metadata for a classification dataset folder.

    Assumes the following structure:

    classification_folder:

        - subfolder_for_label_0:
            - tree_0.nwk
            - tree_1.nwk
            - ...

        - subfolder_for_label_1:
            - tree_0.nwk
            - tree_1.nwk
            - ...

        - ...

    Attributes:
        folder_path (str):
            Path to the classification folder.
        label_folders (List[str]):
            List of subfolder names.
        label_names (List[str]):
            List of label names

    Args:
        folder_path (pathlib.Path):
            Path to the classification folder.
        label_folders (List[str]):
            List of subfolder names.
        label_names (List[str], optional):
            List of label names. Defaults to ``None``. If ``None``, then
            ``label_names`` is set to ``label_folders``.
    """

    def __init__(
        self,
        folder_path: str | pathlib.Path,
        subfolders: List[str],
        label_names: Optional[List[str]] = None,
    ):
        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)
        self.folder_path = folder_path
        self.label_folders = subfolders
        if label_names is None:
            label_names = subfolders
        self.label_names = label_names

    @property
    def num_labels(self) -> int:
        """Number of labels."""
        return len(self.label_folders)

    @property
    def subfolders_paths(self) -> List[pathlib.Path]:
        """List of paths to subfolders."""
        return [
            self.folder_path / subfolder for subfolder in self.label_folders
        ]
