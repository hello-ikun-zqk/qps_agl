from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)
import warnings
import torch
warnings.filterwarnings("ignore")

def get_scores(y, predicted, average="binary"):
    # 确保 y 和 predicted 是 CPU 上的 numpy 数组
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()
    
    accuracy = accuracy_score(y, predicted)
    f1 = f1_score(y, predicted, average=average)
    recall = recall_score(y, predicted, average=average)
    
    return accuracy, f1, recall