import numpy as np

def generate_data(n_samples=20000, std_dev=1.0, seed=42):
    """Создает синтетические данные для обучения"""
    np.random.seed(seed)
    x1 = np.random.normal(0, std_dev, (n_samples, 1))
    x2 = np.random.normal(0, std_dev, (n_samples, 1))
    X_unscaled = np.hstack((x1, x2))
    
    true_w = np.array([[2.0], [0.5]])  # Истинные веса
    y = X_unscaled @ true_w + np.random.normal(0, 3, (n_samples, 1))
    
    return X_unscaled, y, true_w