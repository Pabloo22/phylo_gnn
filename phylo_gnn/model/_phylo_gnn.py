import pytorch_lightning as pl
import torch

from phylo_gnn.model import (
    BaseEncoder,
    BaseMessagePassing,
    BaseReadout,
)


class PhyloGNNModule(pl.LightningModule):
    pass
