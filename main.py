import sam2
import torch
from sam2.build_sam import build_sam2_video_predictor

# get site package path of sam2
import os
import sys
import site
from pathlib import Path

SAM2_CONFIG_DIRECTORY = Path(sam2.__file__).resolve().parent / "configs"
SAM2_CHECKPOINT_DIRECTORY = Path("checkpoints")

from enum import auto, StrEnum


class Sam2Model(StrEnum):
    BASE_PLUS = "b+"
    LARGE = auto()
    SMALL = auto()
    TINY = auto()


def main():
    sam2_model = Sam2Model.BASE_PLUS
    sam2_config_path = SAM2_CONFIG_DIRECTORY / "sam2.1" / f"sam2.1_hiera_{sam2_model}.yaml"
    sam2_checkpoint_path = SAM2_CHECKPOINT_DIRECTORY / f"sam2.1_hiera_{sam2_model.name.lower()}.pt"

    sam2 = build_sam2_video_predictor("/" + str(sam2_config_path), str(sam2_checkpoint_path))


if __name__ == "__main__":
    main()
