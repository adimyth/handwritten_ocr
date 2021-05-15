# Case Independent Handwritten Line Recognition

## Description
The goal of the competition is to create a model that correctly recognizes the handwritten text line present in the image.

## Evaluation
Evaluation Metric will be the Average Levenshtein Distance between the predictions and the ground truth.

## Data
### Raw Data
The data consists of images in "tif" format. Each image has a ground truth text file with the same name. For example, If the image is 1.tif the ground truth file would be 1.gt.txt

The images contain a line written in the english language and the length of the sentence or no. of words can vary. The text values are all in upper case and can also contain special characters.

Data Available Here - https://www.kaggle.com/c/arya-hw-lines/data
![](resources/original.png)

### Processed Data
Sentence level data is converted into word level data. Refer - [Create Word-Level Data Notebook](notebooks/ocr-create-word-level-data.ipynb).

The above notebook differs from the notebook used to split sentences into words for test data because the above notebook utilises training data label. However, this is not available for test data. The test notebook utilises opencv techniques to create splits. Check out [this notebook](notebooks/handwritten-ocr-crnn-v1-inference-part-1.ipynb)

#### Train Data
* [Word Level Train Data - 250x600](https://www.kaggle.com/aditya08/ocr-crnn-train-word-split-250-600)
* [Word Level Train Data - 300x600](https://www.kaggle.com/aditya08/ocr-crnn-train-word-split-300-600)
![](resources/word_level_train_data.png)

#### Test Data
* [Word Level Test Data - 250x600](https://www.kaggle.com/aditya08/ocr-crnn-test-word-split-250-600)
* [Word Level Test Data - 300x600](https://www.kaggle.com/aditya08/ocr-crnn-test-word-split-300-600)
![](resources/word_level_test_data.png)

## Training
1. Download the training data from above & extract it inside `data/train` directory.
2. Run the training script. You can modify the hyperparameters inside the `config.py`.
```bash
poetry run python train.py
```

The data is divided into 3 sections - Train (70%), Valid (15%) & Test (15%). The training script trains model on the train split (`train` method) & evaluates model on all the 3 splits (`infer_all` method).

## Evaluation
Download the test data from above & extract it inside `data/test` directory. To generate submission run the evaluation script.
```bash
poetry run python eval.py
```