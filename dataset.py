import tensorflow as tf  # type: ignore

from config import configs


class OCRDataset:
    def __init__(self, char_to_num):
        super(OCRDataset, self).__init__()
        self.char_to_num = char_to_num

    def encode_single_sample(self, img_path, label):
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
        # 6. Map the characters in label to numbers
        label = self.char_to_num(
            tf.strings.unicode_split(label, input_encoding="UTF-8")
        )
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}

    def get_dataset(self, X, y):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = (
            dataset.map(
                self.encode_single_sample,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .padded_batch(
                configs.BATCH_SIZE
            )  # pads to the smallest per-batch size that fits all elements
            .cache()
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        return dataset
