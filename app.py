import json

import cv2
import numpy as np
import requests
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self):
        super(Predictor, self).__init__()
        self.characters = [
            "]",
            '"',
            "#",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ":",
            ";",
            "?",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "[",
            "!",
        ]
        self.unk_token = "[UNK]"
        self.mask_token = "]"
        # Mapping characters to integers
        char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), num_oov_indices=0, mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=char_to_num.get_vocabulary(), invert=True, mask_token=None
        )

    def encode_single_sample(self, img):
        """
        Processes a single image
        """
        # 1. Resize to the desired size
        img = tf.image.resize(img, [250, 600])
        # 2. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 3. Adds a leading dimension - (1, 600, 250, 1)
        img = np.expand_dims(img, axis=0)
        return {"image": img.tolist(), "label": ""}

    def decode_predictions_greedy(self, pred):
        """
        Greedy Decoding
        """
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = tf.keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=True
        )[0][0][:, : 10]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text


if __name__ == "__main__":
    predictor = Predictor()
    endpoint = "http://localhost:8501/v1/models/handwritten_ocr:predict"
    st.title("Handwritten OCR - TensorFlow Serving")
    st.subheader("Write Something ...")
    canvas_result = st_canvas(
        height=300,
        width=800,
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype("uint8")
        rescaled = cv2.resize(img, (600, 250), interpolation=cv2.INTER_NEAREST)
        st.subheader("Model Input")
        st.image(rescaled)

    if st.button("Predict"):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)    # (300, 800)
        img = img.reshape(img.shape[0], img.shape[1], 1)    # (300, 800, 1)
        input_data = predictor.encode_single_sample(img)

        # Prepare the data that is going to be sent in the POST request
        json_data = json.dumps({"instances": input_data})
        headers = {"content-type": "application/json"}
        # Send the request to the Prediction API
        response = requests.post(endpoint, data=json_data, headers=headers)
        prediction = predictor.decode_predictions_greedy(response.json()["predictions"][0])[0]
        st.success(f"Prediction: {prediction}")