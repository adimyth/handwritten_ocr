import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import Bidirectional  # type: ignore
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Input,
    Layer,
    LeakyReLU,
    MaxPooling2D,
    Reshape,
)
from tensorflow.keras.models import Model  # type: ignore

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


def build_crnn_model():
    """
    Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?
    http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf
    """
    # Inputs to the model
    input_img = Input(
        shape=(configs.IMG_WIDTH, configs.IMG_HEIGHT, 1), name="image", dtype="float32"
    )
    labels = Input(name="label", shape=(None,), dtype="float32")

    # ConvBlock 1 - Conv -> BatchNorm -> LeakyReLU -> MaxPool
    x = Conv2D(16, (3, 3), padding="same", name="conv1")(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    # ConvBlock 2 - Conv -> BatchNorm -> LeakyReLU -> MaxPool
    x = Conv2D(32, (3, 3), padding="same", name="conv2")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    # ConvBlock 3 - Dropout -> Conv -> BatchNorm -> LeakyReLU -> MaxPool
    x = Dropout(0.2)(x)
    x = Conv2D(48, (3, 3), padding="same", name="conv3")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    # ConvBlock 4 - Dropout -> Conv -> BatchNorm -> LeakyReLU
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), padding="same", name="conv4")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # ConvBlock 5 - Dropout -> Conv -> BatchNorm -> LeakyReLU
    x = Dropout(0.2)(x)
    x = Conv2D(80, (3, 3), padding="same", name="conv5")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Columnwise Concatenation
    # 3 MaxPools with pool size and strides 2. Hence feature maps are 8x smaller.
    # Number of filters in last conv block - 80.
    # Reshape accordingly before passing the output to the RNN part of the model
    new_shape = ((configs.IMG_WIDTH // 8), (configs.IMG_HEIGHT // 8) * 80)
    x = Reshape(target_shape=new_shape, name="reshape")(x)

    # RNNBlock 1 - concatenating the output of forward & backward hidden states
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(256, return_sequences=True), merge_mode="concat", name="rnn1"
    )(x)

    # RNNBlock 2
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(256, return_sequences=True), merge_mode="concat", name="rnn2"
    )(x)

    # RNNBlock 3
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(256, return_sequences=True), merge_mode="concat", name="rnn3"
    )(x)

    # RNNBlock 4
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(256, return_sequences=True), merge_mode="concat", name="rnn4"
    )(x)

    # RNNBlock 5
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(256, return_sequences=True), merge_mode="concat", name="rnn5"
    )(x)

    # Linear layer - additional characters for UNK, MASK
    x = Dropout(0.5)(x)
    x = Dense(54 + 3, activation="softmax", name="output")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = tf.keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_crnn"
    )

    # Optimizer
    opt = tf.keras.optimizers.RMSprop(learning_rate=configs.LR)

    # Compile the model and return
    model.compile(optimizer=opt)
    return model


def build_model() -> Model:
    """
    An End-to-End Trainable Neural Network for Image-based Sequence
    Recognition and Its Application to Scene Text Recognition
    https://arxiv.org/pdf/1507.05717.pdf
    """
    # Inputs to the model
    input_img = Input(
        shape=(configs.IMG_WIDTH, configs.IMG_HEIGHT, 1), name="image", dtype="float32"
    )
    labels = Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_1")(input_img)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_2")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_3")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_4")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_5")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)

    # Third conv block
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_6")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_7")(x)
    x = BatchNormalization(name="bn_3")(x)

    # We have used two max pool with pool size and strides of 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing it to RNNs
    new_shape = ((configs.IMG_WIDTH // 4), (configs.IMG_HEIGHT // 4) * 128)
    x = Reshape(target_shape=new_shape, name="reshape")(x)
    x = Dense(64, activation="relu", name="dense1")(x)
    x = Dropout(0.2)(x)

    # RNNs
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(x)

    # Output layer
    x = Dense(55, activation="softmax", name="output")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = Model(inputs=[input_img, labels], outputs=output, name="ocr_model_crnn")
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=configs.LR)
    # Compile the model and return
    model.compile(optimizer=opt)
    return model
