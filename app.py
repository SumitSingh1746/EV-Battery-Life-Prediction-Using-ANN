
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sys
import subprocess
from pathlib import Path

# =====================================
# 1. PAGE CONFIGURATION
# =====================================
st.set_page_config(
    page_title="EV Battery Prediction",
    page_icon="🔋",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "battery_life_model.pkl"
SCALER_PATH = BASE_DIR / "scaler(2).pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# =====================================
# 2. CUSTOM CSS
# =====================================
st.markdown("""
<style>
.main {
    background-color: #f5f8fc;
}
h1 {
    color: #123c69;
}
.stButton > button {
    background-color: #1261a0;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# 3. LOAD SAVED FILES
# =====================================
@st.cache_resource
def load_files():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    return model, scaler, encoder


# =====================================
# 4. APP HEADER
# =====================================
st.title("🔋 EV Battery Prediction")
st.caption("Machine Learning / Deep Learning Prediction App")

st.write(
    "Enter the battery parameters below to generate "
    "a prediction using your trained model."
)

st.divider()

# =====================================
# 5. CHECK FILES AND LOAD MODEL
# =====================================
missing_files = [
    str(path.name)
    for path in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]
    if not path.exists()
]

if missing_files:
    st.error("The following files are missing:")
    for filename in missing_files:
        st.write(f"- `{filename}`")

    st.info(
        "Place app.py and all three .pkl files in the same folder."
    )
    st.stop()

try:
    model, scaler, encoder = load_files()

except ModuleNotFoundError as error:
    st.error(f"Missing Python module: {error.name}")

    st.code(
        f'"{sys.executable}" -m pip install keras tensorflow scikit-learn'
    )

    st.warning(
        "Run the command above in your terminal, then restart Streamlit."
    )
    st.stop()

except Exception as error:
    st.error("Unable to load the saved model files.")
    st.exception(error)

    st.info(
        "The files may require the same library versions used "
        "when they were saved."
    )
    st.stop()

# =====================================
# 6. INPUT FORM
# =====================================
st.subheader("Enter Battery Details")

try:
    categories = encoder.classes_.tolist()
except AttributeError:
    st.error(
        "The label encoder does not have a classes_ attribute. "
        "Check how label_encoder.pkl was created."
    )
    st.stop()

with st.form("battery_prediction_form"):

    battery_type = st.selectbox(
        "Battery Type",
        options=categories
    )

    capacity = st.number_input(
        "Capacity",
        min_value=0.0,
        value=1.6743,
        step=0.01,
        format="%.6f"
    )

    re_value = st.number_input(
        "Re",
        min_value=0.0,
        value=0.0561,
        step=0.001,
        format="%.6f"
    )

    rct_value = st.number_input(
        "Rct",
        min_value=0.0,
        value=0.2010,
        step=0.001,
        format="%.6f"
    )

    predict_button = st.form_submit_button(
        "Predict Now"
    )

# =====================================
# 7. PREDICTION
# =====================================
if predict_button:
    try:
        encoded_type = encoder.transform([battery_type])[0]

        input_df = pd.DataFrame(
            [[encoded_type, capacity, re_value, rct_value]],
            columns=["type", "Capacity", "Re", "Rct"]
        )

        scaled_data = scaler.transform(input_df)

        # Predict with the saved model
        try:
            prediction = model.predict(
                scaled_data, verbose=0
            )
        except TypeError:
            prediction = model.predict(scaled_data)

        result = float(np.asarray(prediction).reshape(-1)[0])

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Ambient Temperature",
            value=f"{result:.2f} °C"
        )

        st.success("Prediction generated successfully.")

        with st.expander("View submitted values"):
            st.dataframe(input_df, use_container_width=True)

        st.caption(
            "The output is an estimate from the trained model. "
            "It represents ambient temperature only if that was "
            "the model's training target."
        )

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)

