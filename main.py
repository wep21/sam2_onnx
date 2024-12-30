import sam2
import torch
from torch import nn
from sam2.build_sam import build_sam2_video_predictor

from pathlib import Path

SAM2_CONFIG_DIRECTORY = Path(sam2.__file__).resolve().parent / "configs"
SAM2_CHECKPOINT_DIRECTORY = Path("checkpoints")

from enum import auto, StrEnum
import argparse


class Sam2Model(StrEnum):
    BASE_PLUS = "b+"
    LARGE = auto()
    SMALL = auto()
    TINY = "t"


class ImageEncoder(nn.Module):
    def __init__(self, sam2_model):
        super().__init__()
        self.sam2_model = sam2_model

    def forward(self, x):
        return self.sam2_model.forward_image(x)


def main():
    parser = argparse.ArgumentParser(description="Export SAM2 model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        choices=[e.value for e in Sam2Model],
        default=Sam2Model.TINY.value,
        help="SAM2 model type",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    args = parser.parse_args()
    sam2_model = Sam2Model(args.model)
    sam2_config_path = (
        SAM2_CONFIG_DIRECTORY / "sam2.1" / f"sam2.1_hiera_{sam2_model}.yaml"
    )
    sam2_checkpoint_path = (
        SAM2_CHECKPOINT_DIRECTORY / f"sam2.1_hiera_{sam2_model.name.lower()}.pt"
    )

    sam2 = build_sam2_video_predictor(
        "/" + str(sam2_config_path), str(sam2_checkpoint_path), args.device
    )

    image_encoder = ImageEncoder(sam2)
    with torch.inference_mode():
        torch.onnx.export(
            image_encoder,
            torch.randn(1, 3, 1024, 1024).to(args.device),
            f"sam2.1_hiera_{sam2_model.name.lower()}_image_encoder.onnx",
            input_names=["input_image"],
            output_names=[
                "vision_features",
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
            ],
            opset_version=args.opset_version,
        )


if __name__ == "__main__":
    main()
