import sys; sys.path.append(".") # fmt: skip
from argparse import ArgumentParser, Namespace

from zipvoice.dataset.datamodule import TtsDataModule


def main():
    parser = ArgumentParser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    
    dm = TtsDataModule(args)
    cuts = dm.train_custom_cuts()
