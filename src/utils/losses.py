import numpy as np

def predict(X_input, w):
    """Предсказание модели"""
    return X_input @ w

def compute_full_loss_no_bn(X_full, y_full, w):
    """Вычисляет функцию потерь на всём наборе данных без BN"""
    m = len(y_full)
    preds = predict(X_full, w)
    return (1/(2*m)) * np.sum((preds - y_full)**2)