import numpy as np
from src.data.data_loader import DataLoader
from src.models.optimizer import compute_batch_gradient_no_bn, compute_bn_forward_and_grad

class SGDTrainer:
    """Класс для обучения моделей с обратной связью для GUI"""
    def __init__(self, X, y, batch_size, initial_w=None, 
                 lr_w=0.01, lr_bn=0.0001, n_epochs=50, epsilon_bn=1e-5):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        
        # Инициализация весов, если не указаны
        if initial_w is None:
            initial_w = np.array([[-8.0], [1.0]])
        self.initial_w = initial_w.copy()
        
        self.lr_w = lr_w
        self.lr_bn = lr_bn
        self.n_epochs = n_epochs
        self.epsilon_bn = epsilon_bn
        
        # Хранение результатов обучения
        self.w_hist_no_bn = None
        self.w_hist_bn = None
        self.w_final_no_bn = None
        self.w_final_bn = None
        
        # Для аниимации
        self.training_in_progress = False
        self.epoch_no_bn = 0
        self.epoch_bn = 0
        self.callback = None
        
    def train_no_bn(self, callback=None):
        """Запуск обучения без Batch Normalization"""
        self.callback = callback
        self.training_in_progress = True
        self.epoch_no_bn = 0
        
        w = self.initial_w.copy()
        w_history = [w.flatten()]
        
        data_loader = DataLoader(self.X, self.y, self.batch_size, shuffle=True)
        
        for epoch in range(self.n_epochs):
            self.epoch_no_bn = epoch + 1
            
            for X_batch, y_batch in data_loader:
                if len(y_batch) == 0:
                    continue
                
                # Градиент без BN
                grad_w = compute_batch_gradient_no_bn(X_batch, y_batch, w)
                
                # Обновляем основные веса
                w = w - self.lr_w * grad_w
                w_history.append(w.flatten())
                
                # Обновляем GUI, если задан callback
                if callback and epoch % 1 == 0:
                    callback(update_no_bn=True)
            
        self.w_final_no_bn = w
        self.w_hist_no_bn = np.array(w_history)
        self.training_in_progress = False
        
        return w, np.array(w_history)
    
    def train_with_bn(self, callback=None, freeze_bn_params=False):
        """Запуск обучения с Batch Normalization"""
        self.callback = callback
        self.training_in_progress = True
        self.epoch_bn = 0
        
        w = self.initial_w.copy()
        w_history = [w.flatten()]
        n_features = self.X.shape[1]
        
        # Инициализация BN параметров
        gamma = np.ones(n_features)
        beta = np.zeros(n_features)
        
        data_loader = DataLoader(self.X, self.y, self.batch_size, shuffle=True)
        
        for epoch in range(self.n_epochs):
            self.epoch_bn = epoch + 1
            
            for X_batch, y_batch in data_loader:
                if len(y_batch) == 0:
                    continue
                
                # Вычисляем градиенты И выход BN
                grad_w, grad_gamma, grad_beta, _ = compute_bn_forward_and_grad(
                    X_batch, y_batch, w, gamma, beta, self.epsilon_bn, compute_grads=True
                )
                
                # Обновляем gamma и beta ТОЛЬКО если они не заморожены
                if not freeze_bn_params:
                    gamma = gamma - self.lr_bn * grad_gamma
                    beta = beta - self.lr_bn * grad_beta
                
                # Обновляем основные веса
                w = w - self.lr_w * grad_w
                w_history.append(w.flatten())
                
                # Обновляем GUI, если задан callback
                if callback and epoch % 1 == 0:
                    callback(update_bn=True)
            
        self.w_final_bn = w
        self.w_hist_bn = np.array(w_history)
        self.training_in_progress = False
        
        return w, np.array(w_history)
    
    def get_histories(self):
        """Возвращает истории обучения для визуализации"""
        return {
            'no_bn': self.w_hist_no_bn,
            'with_bn': self.w_hist_bn,
            'initial_w': self.initial_w,
            'true_w': np.array([[2.0], [0.5]])
        }
    
    def get_final_weights(self):
        """Возвращает финальные веса для отображения"""
        return {
            'no_bn': self.w_final_no_bn,
            'with_bn': self.w_final_bn,
            'true_w': np.array([[2.0], [0.5]])
        }