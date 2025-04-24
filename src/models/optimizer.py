import numpy as np
from src.utils.losses import predict

def compute_batch_gradient_no_bn(X_batch, y_batch, w):
    """Градиент БЕЗ BN"""
    m_batch = len(y_batch)
    preds = predict(X_batch, w)
    dLoss_dPred = (preds - y_batch) / m_batch
    return X_batch.T @ dLoss_dPred

def compute_bn_forward_and_grad(X_batch, y_batch, w, gamma, beta, epsilon=1e-5, compute_grads=True):
    """Функция градиента С BN"""
    m_batch = len(y_batch)
    if m_batch == 0:
        return np.zeros_like(w), np.zeros_like(gamma), np.zeros_like(beta), None

    # --- Прямой проход BN ---
    batch_mean = np.mean(X_batch, axis=0, keepdims=True)
    batch_var = np.var(X_batch, axis=0, keepdims=True)
    batch_std_inv = 1.0 / np.sqrt(batch_var + epsilon)
    X_normalized = (X_batch - batch_mean) * batch_std_inv
    X_bn_output = X_normalized * gamma + beta

    if not compute_grads:
        return None, None, None, X_bn_output

    # --- Предсказание и градиент dL/dPred ---
    predictions = predict(X_bn_output, w)
    dLoss_dPred = (predictions - y_batch) / m_batch

    # --- Обратный проход ---
    grad_w = X_bn_output.T @ dLoss_dPred

    # Градиенты по параметрам BN
    dLoss_dXbn = dLoss_dPred @ w.T
    grad_beta = np.sum(dLoss_dXbn, axis=0)
    grad_gamma = np.sum(dLoss_dXbn * X_normalized, axis=0)

    return grad_w, grad_gamma, grad_beta, X_bn_output