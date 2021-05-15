import ast

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from config import configs


class OCRCRNNPredictor:
    def __init__(self):
        super(OCRCRNNPredictor, self).__init__()
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
        model = tf.keras.models.load_model(f"{configs.EXP_NAME}_model_checkpoints/")
        self.prediction_model = tf.keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )

    def encode_single_sample(self, img_path):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [configs.IMG_HEIGHT, configs.IMG_WIDTH])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Return img
        return img

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = tf.keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=True
        )[0][0][:, : configs.MAX_LENGTH]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def get_all_preds(test_df):
        """
        Utility function that returns both model prediction & actual labels
        """
        labels = []
        for idx, row in tqdm(test_df.iterrows()):
            row_pred = []
            image_splits = [
                f"data/test/ocr-crnn-test-word-split-250-600/test_images/test_images/{x}.png"
                for x in ast.literal_eval(row["Splits"])
            ]
            for img_split in image_splits:
                processed_split = np.expand_dims(
                    self.encode_single_sample(img_split), axis=0
                )
                split_pred = self.prediction_model.predict(processed_split)
                split_pred = self.decode_batch_predictions(split_pred)[0]
                # handling UNK & MASK token
                split_pred = split_pred.replace(self.unk_token, "").replace(
                    self.mask_token, ""
                )
                row_pred.append(split_pred)
            labels.append(" ".join(row_pred))
        return labels

    def make_submission(self):
        test_df = pd.read_csv("data/ocr-crnn-test-word-split-250-600/test_csv.csv")
        predictions = self.get_all_preds(test_df)
        submission_df = pd.DataFrame.from_dict(
            {
                "imageName": test_df["Path"].apply(lambda x: x.split("/")[-1]),
                "prediction": predictions,
            }
        )
        submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    predictor = OCRCRNNPredictor()
    predictor.make_submission()
