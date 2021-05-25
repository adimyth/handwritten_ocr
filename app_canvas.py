import json

import cv2 # type: ignore
import numpy as np # type: ignore
import requests
import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
from streamlit_drawable_canvas import st_canvas # type: ignore


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
        st.write(f"Resized Image: {img.numpy().shape}")
        # 2. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        st.write(f"Encoded Image: {img.numpy().shape}")
        return {"image": img.numpy().tolist(), "label": 0}

    def decode_predictions(self, pred):
        """
        Greedy Decoding
        """
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = tf.keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=True
        )[0][0][:, :10]
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
        stroke_width=5,
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
        st.write(f"Canvas Image: {img.shape}")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  # (300, 800)
        st.write(f"Greyscaled Image: {img.shape}")
        img = img.reshape(img.shape[0], img.shape[1], 1)  # (300, 800, 1)
        st.write(f"Psedudo Dimension: {img.shape}")
        input_data = np.expand_dims(predictor.encode_single_sample(img), axis=0)

        # Prepare the data that is going to be sent in the POST request
        json_data = json.dumps({"instances": input_data.tolist()})
        headers = {"content-type": "application/json"}
        # Send the request to the Prediction API
        response = requests.post(endpoint, data=json_data, headers=headers)
        interim = np.array(response.json()["predictions"][0])
        prediction = predictor.decode_predictions(np.expand_dims(interim, axis=0))[0]
        prediction = prediction.replace(predictor.unk_token, "").replace(
            predictor.mask_token, ""
        )
        st.success(f"Prediction: {prediction}")
