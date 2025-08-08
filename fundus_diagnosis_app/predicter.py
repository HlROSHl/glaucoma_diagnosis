from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import onnxruntime as rt  # type: ignore
from fundus_diagnosis_app.cropper import FundusImageCropping


def get_resource_path(relative_path: Path) -> Path:
    if hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parents[1]

    return base_path / relative_path


@dataclass
class PredictorInput:
    image: np.ndarray
    color_mode: Literal["RGB", "BGR"]


class GlaucomaPredictor:
    default_model_path = get_resource_path(
        Path("models/glaucomaUCSDExperiment_ResNet_20190516_192102/model.onnx")
    )

    def __init__(
        self,
        session: rt.InferenceSession,
        cropper: FundusImageCropping | None = None,
    ) -> None:
        self.cropper = cropper
        self.session = session
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @classmethod
    def default(cls, cropper: FundusImageCropping | None = None) -> GlaucomaPredictor:
        session = rt.InferenceSession(str(cls.default_model_path))
        return cls(session, cropper)

    def predict_from_optic_disk(
        self, input: PredictorInput
    ) -> tuple[list[float], float]:
        input_tensor = self.transform_optic_disk_image(input)
        result = self.session.run([self.output_name], {self.input_name: input_tensor})[
            0
        ][0]
        softmax = np.exp(result) / np.sum(np.exp(result))
        return result, softmax[1]

    def predict(self, input: PredictorInput) -> tuple[list[float], float]:
        if self.cropper is None:
            raise ValueError("cropper is not set")

        optic_disk_image = self.cropper.get_opticdisk(input.image, resize_to=(224, 224))
        return self.predict_from_optic_disk(
            PredictorInput(optic_disk_image, input.color_mode)
        )

    def transform_optic_disk_image(
        self, optic_disk_image_input: PredictorInput
    ) -> np.ndarray:
        # 1. Equivalent to torchvision.transforms.ToTensor()
        # - scale image to [0, 1]
        # - convert input image's shape: (H, W, C) into (1, C, H, W)
        fundus_image = optic_disk_image_input.image
        tensor = (
            np.transpose(fundus_image / 255.0, (2, 0, 1))
            .reshape(1, 3, 224, 224)
            .astype(np.float32)
        )
        # 2. Convert BGR to RGB
        if optic_disk_image_input.color_mode == "BGR":
            tensor = tensor[:, ::-1, :, :]

        # 3. Equivalent to torchvision.transforms.Normalize(mean=[0.292, 0.466, 0.763], std=[0.124, 0.140, 0.161])
        mean = np.array([0.292, 0.466, 0.763]).reshape(1, 3, 1, 1)
        std = np.array([0.124, 0.140, 0.161]).reshape(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.astype(np.float32)
