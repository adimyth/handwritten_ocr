import os
import random
from glob import glob
from itertools import chain

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import Model  # type: ignore
from tqdm import tqdm  # type: ignore

from config import configs
from dataset import OCRDataset
from metrics import get_mean_cer_score, get_mean_lev_score
from model import build_crnn_model
from utils import plot_hist


class OCRCRNNTrainer:
    def __init__(self):
        super(OCRCRNNTrainer, self).__init__()
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

        # SEED Everything
        seed = configs.SEED
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # Mapping characters to integers
        self.char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), num_oov_indices=0, mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), invert=True, mask_token=None
        )
        self.dataset = OCRDataset(self.char_to_num)
        self.model = build_crnn_model()

    def load_df(self):
        df = pd.read_csv("data/train/train.csv")[["Path", "Labels"]]
        df["Path"] = df["Path"].apply(lambda x: f"data/train/images/{x}")
        df = df.loc[df["Labels"].str.len() <= configs.MAX_LENGTH]
        print(f"Total Data: {df.shape}")
        return df

    def split_data(self, df):
        X_train, X_valid, y_train, y_valid = train_test_split(
            df["Path"],
            df["Labels"],
            random_state=configs.SEED,
            stratify=df["Labels"].str.len(),
            test_size=0.3,
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_valid, y_valid, test_size=0.5, random_state=configs.SEED
        )
        print(f"Training Data: {X_train.shape, y_train.shape}")
        print(f"Validation Data: {X_valid.shape, y_valid.shape}")
        print(f"Testing Data: {X_test.shape, y_test.shape}")
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def load_data(self):
        df = self.load_df()
        X_train, y_train, X_valid, y_valid, X_test, y_test = self.split_data(df)
        self.train_dataset = self.dataset.get_dataset(X_train, y_train)
        self.validation_dataset = self.dataset.get_dataset(X_valid, y_valid)
        self.test_dataset = self.dataset.get_dataset(X_test, y_test)

    def train(self):
        self.load_data()
        # EarlyStopping
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        # ModelCheckpoint
        model_checkpoint = ModelCheckpoint(
            filepath=f"{configs.EXP_NAME}_model_checkpoint",
            save_weights_only=False,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )
        # CSVLogger
        csv_logger = CSVLogger(f"{configs.EXP_NAME}_training.log", separator="\t")

        # Training
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=configs.EPOCHS,
            callbacks=[early_stopping, model_checkpoint, csv_logger],
        )
        self.model.save(f"{configs.EXP_NAME}_model.h5")
        plot_hist(history)

    def decode_batch_predictions_greedy(self, pred):
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

    def decode_batch_predictions_beamsearch(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = tf.keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=False, beam_width=10, top_k_paths=1
        )[0][0][:, : configs.MAX_LENGTH]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def get_all_preds(self, dataset, prediction_model, greedy=True):
        """
        Utility function that returns both model prediction & actual labels
        """
        decoded_text, actuals_text = [], []
        for batch in tqdm(dataset):
            batch_images = batch["image"]
            batch_labels = batch["label"]

            preds = prediction_model.predict(batch_images)
            if greedy:
                pred_texts = self.decode_batch_predictions_greedy(preds)
            else:
                pred_texts = self.decode_batch_predictions_beamsearch(preds)
            # handling UNK & MASK token
            pred_texts = [
                x.replace(self.unk_token, "").replace(self.mask_token, "")
                for x in pred_texts
            ]

            orig_texts = []
            for label in batch_labels:
                label = (
                    tf.strings.reduce_join(self.num_to_char(label))
                    .numpy()
                    .decode("utf-8")
                )
                orig_texts.append(
                    label.replace(self.unk_token, "").replace(self.mask_token, "")
                )

            decoded_text.append(pred_texts)
            actuals_text.append(orig_texts)

        # flatten 2D list
        decoded_text = list(chain.from_iterable(decoded_text))
        actuals_text = list(chain.from_iterable(actuals_text))
        return decoded_text, actuals_text

    def infer_all(self):
        self.prediction_model = Model(
            self.model.get_layer(name="image").input,
            self.model.get_layer(name="output").output,
        )
        train_decoded_text, train_actuals_text = self.get_all_preds(
            self.train_dataset, self.prediction_model
        )
        print(
            f"Training Accuracy: {accuracy_score(train_actuals_text, train_decoded_text):.5f}"
        )
        print(
            f"Training Mean Levenshtein Distance: {get_mean_lev_score(train_actuals_text, train_decoded_text):.5f}"
        )
        print(
            f"Training Mean CER Score: {get_mean_cer_score(train_actuals_text, train_decoded_text):.5f}"
        )

        valid_decoded_text, valid_actuals_text = self.get_all_preds(
            self.validation_dataset, self.prediction_model
        )
        print(
            f"Validation Accuracy: {accuracy_score(valid_actuals_text, valid_decoded_text):.5f}"
        )
        print(
            f"Validation Mean Levenshtein Distance: {get_mean_lev_score(valid_actuals_text, valid_decoded_text):.5f}"
        )
        print(
            f"Validation Mean CER Score: {get_mean_cer_score(valid_actuals_text, valid_decoded_text):.5f}"
        )

        test_decoded_text, test_actuals_text = self.get_all_preds(
            self.test_dataset, self.prediction_model
        )
        print(
            f"Test Accuracy: {accuracy_score(test_actuals_text, test_decoded_text):.5f}"
        )
        print(
            f"Test Mean Levenshtein Distance: {get_mean_lev_score(test_actuals_text, test_decoded_text):.5f}"
        )
        print(
            f"Test Mean CER Score: {get_mean_cer_score(test_actuals_text, test_decoded_text):.5f}"
        )


if __name__ == "__main__":
    trainer = OCRCRNNTrainer()
    trainer.train()
    trainer.infer_all()
