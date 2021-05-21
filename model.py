import tensorflow as tf # type: ignore
from tensorflow.keras.layers import (LSTM, BatchNormalization, Bidirectional, # type: ignore
                                     Conv2D, Dense, Dropout, Input, Layer,
                                     MaxPooling2D, Reshape)
from tensorflow.keras.models import Model # type: ignore

from config import configs


class CTCLayer(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_crnn_model() -> Model:
    """
    Returns a CRNN model with CTC layer at the end
    """
    # Inputs to the model
    input_img = Input(
        shape=(configs.IMG_WIDTH, configs.IMG_HEIGHT, 1), name="image", dtype="float32"
    )
    labels = Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_1")(input_img)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_2")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_3")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_4")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="conv_5")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)

    # Third conv block
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="conv_6")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="conv_7")(x)
    x = BatchNormalization(name="bn_3")(x)

    # We have used two max pool with pool size and strides of 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing it to RNNs
    new_shape = ((configs.IMG_WIDTH // 4), (configs.IMG_HEIGHT // 4) * 256)
    x = Reshape(target_shape=new_shape, name="reshape")(x)
    x = Dense(128, activation="relu", name="dense1")(x)
    x = Dropout(0.2)(x)

    # RNNs
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)

    # Output layer
    x = Dense(55, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = Model(inputs=[input_img, labels], outputs=output, name="ocr_model_crnn")
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=configs.LR)
    # Compile the model and return
    model.compile(optimizer=opt)
    return model
