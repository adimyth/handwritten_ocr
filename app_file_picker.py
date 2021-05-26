import json

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import requests
import streamlit as st  # type: ignore
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore


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
        # 1. Read image
        img = tf.io.read_file("temp.png")
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [250, 600])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        return img.numpy().tolist()

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
            # Hack - res+1 below due to shift in char-num mapping. [UNK] token is responsible
            res = (
                tf.strings.reduce_join(self.num_to_char(res + 1))
                .numpy()
                .decode("utf-8")
            )
            output_text.append(res)
        return output_text


if __name__ == "__main__":
    predictor = Predictor()

    # file uploader
    st.title("Handwritten OCR - TensorFlow Serving")
    uploaded_file = st.file_uploader(label="Upload Image", type=["jpg", "png"])

    # sidebar
    version = st.sidebar.selectbox("Select Version", ("Version 1", "Version 2"))
    endpoint = f"http://localhost:8501/v1/models/handwritten_ocr/versions/{version.split()[1]}:predict"

    if uploaded_file:
        img = Image.open(uploaded_file)  # (w, h)
        st.image(img, caption=f"{uploaded_file.name} - {img.size}")

    if st.button("Predict"):
        if uploaded_file is None:
            st.error("Select an image")
        else:
            # saving image temporarily
            plt.imshow(img.convert("RGB"))  # saving as RGB
            plt.axis("off")
            plt.savefig("temp.png")
            input_data = np.expand_dims(predictor.encode_single_sample(img), axis=0)

            # Prepare the data that is going to be sent in the POST request
            json_data = json.dumps({"instances": input_data.tolist()})
            headers = {"content-type": "application/json"}
            # Send the request to the Prediction API
            response = requests.post(endpoint, data=json_data, headers=headers)

            interim = np.array(response.json()["predictions"][0])
            prediction = predictor.decode_predictions(np.expand_dims(interim, axis=0))[
                0
            ]
            prediction = prediction.replace("[UNK]", "").replace("]", "")
            st.success(f"Prediction: {prediction}")
