# [Case Independent Handwritten Line Recognition](https://www.kaggle.com/c/arya-hw-lines)

## Description
The goal of the competition is to create a model that correctly recognizes the handwritten text line present in the image.

## Evaluation
Evaluation Metric will be the Average Levenshtein Distance between the predictions and the ground truth.

## Raw Data
The data consists of images in "tif" format. Each image has a ground truth text file with the same name. For example, If the image is 1.tif the ground truth file would be 1.gt.txt

The images contain a line written in the english language and the length of the sentence or no. of words can vary. The text values are all in upper case and can also contain special characters.

Data Available Here - https://www.kaggle.com/c/arya-hw-lines/data

![](resources/original.png)

## Data Processing
### Train Data
Sentence level data is converted into word level data. Refer - [Create Word-Level Data Notebook](notebooks/create_word_level_data.ipynb).

* [Word Level Train Data - 250x600](https://www.kaggle.com/aditya08/ocr-crnn-train-word-split-250-600)
* [Word Level Train Data - 300x600](https://www.kaggle.com/aditya08/ocr-crnn-train-word-split-300-600)
![](resources/word_level_train_data.png)

### Test Data
Since labels are not available for test data, we cannot use the above technique. So for test data we utilise opencv techniques to create splits. Code taken from [stackoverflow post](https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv). Check out [this notebook](notebooks/handwritten-ocr-crnn-v1-inference-part-1.ipynb) to know more


* [Word Level Test Data - 250x600](https://www.kaggle.com/aditya08/ocr-crnn-test-word-split-250-600)
* [Word Level Test Data - 300x600](https://www.kaggle.com/aditya08/ocr-crnn-test-word-split-300-600)
![](resources/word_level_test_data.png)

## [Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?](http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf)

### Architecture
**Convolutional Blocks**: 5 convolution blocks with each block containing a Conv2D layer, with `3x3` kernel size & stride 1. The number of filters at the `n-th` Conv layer is to `16n`. Dropout is applied at the input of last 2 conv blocks (prob=0.2). Batch Normalization is used to normalize the inputs to the nonlinear activation function. *LeakyReLU* is the activation function in the convolutional blocks. Finally, *Maxpool* with non-overlapping kernels
of `2×2` is applied.

**Recurrent Blocks**: Recurrent blocks are formed by bidirectional 1D-LSTM layers, that process the input image columnwise in left-to-right and right-to-left order. The output of the two directions
is concatenated depth-wise. Dropout is also applied(prob=0.5). Number of hidden units in all LSTM layers `256`. Total number of recurrent blocks is 5.

**Linear Layer**:  Finally, each column after the recurrent 1D-LSTM blocks must be mapped to an output label. The depth is transformed from `2D` to `L` using an affine transformation (L=characters+1)

### Parameters
1. RMSProp with learning rate - 0.0003
2. Batch Size = 16

### Augmentation
Rotation, Translation, Scaling and Shearing (all performed as a single affine transform) and gray-scale erosion and dilation. Each of these operations is applied dynamically and independently on
each image of the training batch (each with 0.5 probability). Thus, the exact same image is virtually never observed twice during training.

![](https://imgur.com/Hrum8pX.png)

## [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf)
CRNN model consists of CNN & RNN blocks along with a Transcription Layer.
* CNN Block - Three convolution blocks (7 conv layers) & maxpool layer. Extracts features from image.
* RNN Block - Two Bidirectional LSTM layers. Splits features into some feature sequences & pass it to recurrent layers
* Transcription Layer - Conversion of Feature-specific predictions to Label using CTC. CTC loss is specially designed to optimize both the *length* of the predicted sequence and the *classes* of the predicted sequence

In CRNN convolution feature maps are transformed into a sequence of feature vectors. It is then fed to LSTM/GRU which produces a probability distribution for each feature vector and each label. For example, consider the output of CNN Block is - `(batch_size, 64, 4, 32)` where dimensions are `(batch_size, channels, height, width)`. Then we need to permute dimensions to `(batch_size, width, height, channels)` so that channels is the last one.

> Each feature vector of a feature sequence is generated from left to right on the feature maps by column. This means the i-th feature vector is the concatenation of the i-th columns of all the maps.

It is then reshaped to `(batch_size, 32, 256)` and fed into the GRU layers. GRU produces tensor of shape `(batch_size, 32, 256)` which is passed through fully-connected layer and log_softmax function to return tensor of the shape `(batch_size, 32, vocabulary)`. **This tensor for each image in the batch contains probabilities of each label for each input feature.**

![CRNN](https://miro.medium.com/max/894/0*nGWtig3Cd0Jma2nX)

## Training
Trains a CRNN model at the word level. Saves the best model based on validation loss. Supports **Greedy** & **Beam Search** Decoding. Reports the following metrics on both training & validation data -
1. Accuracy
2. Mean Levenshtein Distance
3. Character Error Rate

The data is divided into 3 sections - Train (70%), Valid (15%) & Test (15%). The training script trains model on the train split (`train` method) & evaluates model on all the 3 splits (`infer_all` method).

### Running it locally
1. Download the training data from above & extract it inside `data/train` directory. Some sample data is available already
2. Run the training script. You can modify the hyperparameters inside the `config.py`.
```bash
poetry run python train.py
```

## Evaluation
Download the test data from above & extract it inside `data/test` directory. To generate submission run the evaluation script.
```bash
poetry run python eval.py
```
Supports *Greedy* as well *Beam Search Decoding* based on choice. Set `greedy=False` in `make_submission` function for BeamSearch Decoding.

## Result
The first architecture described above performed better than the second one. Once that was decided I ran multiple experiments with varying degrees of image size, model depth, number of epoch, etc. The following configuration worked the best -

* BATCH_SIZE - 16
* EPOCHS - 20
* IMG_HEIGHT - 250
* IMG_WIDTH - 600
* MAX_LENGTH - 10

### Word Level - Training & Validation Loss
![](resources/losses.png)

### Test Data Prediction
![](resources/predictions.png)

### Word Level Metrics
| Metric | Training | Validation |
--- | --- | ---
|Accuracy|0.8162|0.7392|
|Levenhstein Distance|0.3192|0.5001|
|Character Error Rate|7.2515|11.3602|

### Sentence Level Metrics
* Accuracy - 0.326 (Surprising?)
* Levenhstein Distance - 2.15
* Character Error Rate - 6.77

### Kaggle LeaderBoard
![](resources/leaderboard.png)

* Next Best Private LB - 4.54741
* Next Best Public LB - 4.29530

## What didn't work
1. Training for longer epochs didn't work. All of the runs EarlyStopped
2. Centering the image & center cropping didn't work
3. Larger Image size - 350x800 performed poorly then 250x600
4. Increasing number of characters from 8 to 10 gave better scores, however at 12, the score got poorer
5. Most of the time leaderboard score for GreedySearch was better than BeamSearch

## Next Steps
1. Add Spatial Transformer Network Component
2. https://arxiv.org/pdf/1904.09150.pdf
3. https://arxiv.org/abs/2012.04961
