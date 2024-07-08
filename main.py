"""
This file contains a Streamlit application for computing the net income breakdown for Filipinos.

The application allows users to input their monthly basic income, non-taxable allowance, and choose
whether to include night differential rate. It then calculates the net income and provides a
breakdown of the income components including basic salary, allowance, night differential, gross
income, employee contributions, monthly tax, and monthly net income.
"""

from datetime import datetime

import cv2
import keras
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from config.site_config import div_configs
from utils.utils import SessionState

## ======================================================================================= ##
## SITE TITLE
TITLE = "Ovarian Ultrasound Image Analysis Using Deep Learning"
ICON = "🏥"
QUICK_DESCRIPTION = ""
# QUICK_DESCRIPTION = "Ovarian Ultrasound Image Analysis Using Deep Learning"

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title=TITLE,
    page_icon=ICON,
)


@st.cache_resource
def load_model(path):
    """
    Load a Keras model from the given path.

    Parameters:
    path (str): The path to the saved model file.

    Returns:
    keras.models.Model: The loaded Keras model.
    """
    loaded_model = keras.models.load_model(path)
    return loaded_model


model = load_model("model/efficientnetV2B0_trained.keras")


## ======================================================================================= ##
## SITE CONFIGURATION

# st.markdown(div_configs["hide_github_icon"], unsafe_allow_html=True)
# st.markdown(div_configs["hide_streamlit_style"], unsafe_allow_html=True)


## ======================================================================================= ##
## SITE CONTENTS


title = st.container()
features = st.expander("Features", expanded=False)
contents = st.container()
results = st.container()

button = SessionState(key="button_clicked", default_state=False)

with title:
    st.title(f" {ICON} {TITLE}")
    if QUICK_DESCRIPTION != "":
        st.subheader(QUICK_DESCRIPTION)

    CLASS_NAMES = ["Benign", "Malignant"]


# with contents:
#     st.subheader("Upload Image/s:")
#     st.markdown(
#         """
#         #### Upload up to 5 grayscale images here:
#     """
#     )
#     grayscale_images = st.file_uploader(
#         "Choose a .jpg or .png file", accept_multiple_files=True, key="grayscale_images"
#     )

#     st.markdown(
#         """
#         #### Upload up to 5 color doppler images here:
#     """
#     )
#     doppler_images = st.file_uploader(
#         "Choose a .jpg or .png file", accept_multiple_files=True, key="doppler_images"
#     )


with contents:
    st.subheader("Upload Image/s:")
    ovarian_images = st.file_uploader(
        "Choose a .jpg or .png file",
        accept_multiple_files=True,
        type=["png", "jpg"],
        on_change=button.sessionstate_false,
    )

    # IF no file is uploaded
    if ovarian_images is None or ovarian_images == []:
        st.write("Please upload the file")
    else:
        st.write("File/s has uploaded succesfully")

    evaluate_button = st.button(
        label="Evaluate",
        type="primary",
        on_click=button.sessionstate_true,
        use_container_width=True,
    )


with results:
    if evaluate_button or button.check_sessionstate():
        st.subheader("Result")

        images_to_be_evaluated = {}
        init_image_num = 1
        for image in ovarian_images:
            bytes_data = image.getvalue()
            image = tf.image.decode_image(bytes_data)[..., :3]
            resized_image = tf.image.resize(image, (224, 224))
            image_array = tf.expand_dims(resized_image, 0)
            prediction = model.predict(image_array, verbose=0)
            predicted_value = CLASS_NAMES[np.argmax(prediction)]
            images_to_be_evaluated[init_image_num] = {
                "predicted_value": predicted_value,
                "proba_0": prediction[0][0],
                "proba_1": prediction[0][1],
            }
            init_image_num += 1

        results_df = pd.DataFrame(images_to_be_evaluated).T

        st.write()

        count_1st_class = results_df["predicted_value"].value_counts()[0]
        count_2nd_class = (
            results_df["predicted_value"].value_counts()[1]
            if len(results_df["predicted_value"].value_counts()) > 1
            else 0
        )

        if count_1st_class == count_2nd_class:
            results_df_grouped = results_df.groupby(
                ["predicted_value"], as_index=False
            ).agg(
                predicted_counts=("predicted_value", "count"),
                average_pred_0=("proba_0", "mean"),
                average_pred_1=("proba_1", "mean"),
            )
            results_df_grouped = results_df_grouped.agg(
                Benign=("average_pred_0", "max"),
                Malignant=("average_pred_1", "max"),
            )

            benign_value = results_df_grouped["average_pred_0"].values[0]
            malignant_value = results_df_grouped["average_pred_1"].values[1]

            if benign_value > malignant_value:
                overall_pred = "Benign"
            else:
                overall_pred = "Malignant"
        else:
            overall_pred = results_df["predicted_value"].value_counts().idxmax()

        prediction_text_color = "green" if overall_pred == "Benign" else "red"
        st.markdown(
            f"""
        ##### {len(ovarian_images)} {'image was' if len(ovarian_images) == 1 else 'images were'}  uploaded, overall prediction is: :{prediction_text_color}[**{overall_pred.upper()}**]
        """
        )

        st.write(results_df)

        ## Generating heatmap

        index_pred = list(
            results_df[results_df["predicted_value"] == overall_pred].index
        )
        index_pred = [value - 1 for value in index_pred]

        for predicted_images_index in index_pred:
            st.image(ovarian_images[predicted_images_index], width=500)
