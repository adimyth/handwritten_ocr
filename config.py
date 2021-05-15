from attrdict import Attrdict

configs = {
    "SEED": 42,
    "IMG_HEIGHT": 250,
    "IMG_WIDTH": 600,
    "EPOCHS": 8,
    "BATCH_SIZE": 16,
    "LR": 0.001,
    "MAX_LENGTH": 8,
    "EXP_NAME": "crnn_v1",
}
configs = Attrdict(configs)
