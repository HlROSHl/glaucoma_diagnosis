from __future__ import annotations
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import onnxruntime as rt  # type: ignore
import pandas as pd  # type: ignore
import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from fundus_diagnosis_app.cropper import FundusImageCropping
from fundus_diagnosis_app.predicter import GlaucomaPredictor, PredictorInput


class PredictionResult(TypedDict):
    probability_of_glaucoma: float
    filename: str


def get_system_dir() -> Path:
    """
    Create and/or return the local user data folder: ~/.fundus_diagnosis_app
    """
    system_dir = Path.home() / ".fundus_diagnosis_app"
    system_dir.mkdir(parents=True, exist_ok=True)
    return system_dir


def save_results(results: list[PredictionResult]) -> str:
    """
    Saves prediction results as a CSV in the user data folder.
    Returns the filepath of the saved CSV.
    """
    system_dir = get_system_dir()
    timestamp = int(time.time())
    result_file = system_dir / f"prediction-{timestamp}.csv"
    df = pd.DataFrame(results).loc[:, ["filename", "probability_of_glaucoma"]]
    df.to_csv(result_file, index=False)
    return str(result_file)


def predict(
    f: UploadedFile, sess: rt.InferenceSession, no_crop: bool = False
) -> PredictionResult:
    st.header("Score")
    placeholder_score = st.empty()
    st.divider()
    st.text("Model input (Optic disk is cropped)")
    placeholder_crop = st.empty()

    with st.spinner("Cropping optic disk..."):
        # Load the image
        imagep = Image.open(f)
        image = np.array(imagep.convert("RGB"))[:, :, ::-1]

        # Predict
        cropper = FundusImageCropping(img_shape=image.shape)
        predictor = GlaucomaPredictor(sess, cropper)
        if no_crop:
            cropped = image
        else:
            cropped = cropper.get_opticdisk(image, resize_to=224)

        placeholder_crop.image(cropped, caption="Cropped Image.", channels="BGR")

    with st.spinner("Classifying..."):
        _, prediction = predictor.predict_from_optic_disk(
            PredictorInput(cropped, "BGR")
        )
        placeholder_score.subheader(f"Glaucoma: {prediction * 100:.4f} %")

    return {"probability_of_glaucoma": prediction, "filename": f.name}


@st.cache_resource
def load_model() -> rt.InferenceSession:
    """Loads the ONNX model once and caches it."""
    return rt.InferenceSession(GlaucomaPredictor.default_model_path)


# ------------
# MAIN SCRIPT
# ------------
st.set_page_config(page_title="Fundus Diagnosis App - Prediction", layout="wide")
st.title("Fundus Diagnosis App - Prediction")

# Load model (cached)
session = load_model()

uploaded_files = st.file_uploader(
    "Choose image(s) for prediction",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and len(uploaded_files) > 0:
    prediction_results: list[PredictionResult] = []

    finish_message = st.empty()
    for idx, f in enumerate(uploaded_files):
        expanded = True if idx == 0 else False
        with st.expander(f.name, expanded=expanded):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.image(f, caption="Uploaded Image.")

            with col2:
                prediction = predict(f, session)
                prediction_results.append(prediction)

    # Save all results
    result_file = save_results(prediction_results)
    finish_message.success(f"Prediction finished. Results are saved to {result_file}")
