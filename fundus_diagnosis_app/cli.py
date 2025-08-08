import argparse
import time
from pathlib import Path

import cv2
import pandas as pd  # type: ignore

from fundus_diagnosis_app.cropper import FundusImageCropping
from fundus_diagnosis_app.predicter import GlaucomaPredictor, PredictorInput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="input directory")
    parser.add_argument(
        "--output",
        type=str,
        # default=f"predict-{time.time()}.csv",
        default="predict.csv",
        help="output file",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="do not crop the image, use the whole image instead",
    )
    args = parser.parse_args()

    files = list(Path(args.input_dir).rglob("**/*.jpg"))
    predictor = GlaucomaPredictor.default()
    result = []
    for file in files:
        img = cv2.imread(str(file))
        if not args.no_crop:
            cropper = FundusImageCropping(img_shape=img.shape)
            cropped = cropper.get_opticdisk(img, resize_to=224)
        else:
            cropped = img
        scores, prediction = predictor.predict_from_optic_disk(
            PredictorInput(cropped, "BGR")
        )
        result.append(
            {
                "filename": file.name,
                "raw_output_normal": scores[0],
                "raw_output_glaucoma": scores[1],
                "prob": prediction,
            }
        )
    df = pd.DataFrame(result)
    df.to_csv(str(args.output), index=False)
