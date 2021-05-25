import fastwer  # type: ignore
import Levenshtein  # type: ignore
import numpy as np  # type: ignore


def get_mean_lev_score(y_true: list, y_pred: list):
    """
    Returns mean Levenshtein Distance.
    """
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score


def get_mean_cer_score(y_true: list, y_pred: list):
    """
    Returns CER Score for the Corpus.
    """
    return fastwer.score(y_pred, y_true, char_level=True)
