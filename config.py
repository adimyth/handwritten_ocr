from attrdict import AttrDict  # type: ignore

configs = {
    "SEED": 42,
    "IMG_HEIGHT": 250,
    "IMG_WIDTH": 600,
    "EPOCHS": 1,
    "BATCH_SIZE": 16,
    "LR": 0.0003,
    "MAX_LENGTH": 8,
    "EXP_NAME": "ocr_crnn",
}
configs = AttrDict(configs)
