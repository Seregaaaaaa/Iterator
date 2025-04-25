import numpy as np
from abc import ABC, abstractmethod
from src.data.data_loader import DataLoader
from src.models.optimizer import compute_batch_gradient_no_bn, compute_bn_forward_and_grad

class AbstractTrainer(ABC):
    """Абстрактный базовый класс для обучения моделей с применением паттерна Шаблонный метод"""
    def __init__(self, X, y, batch_size, initial_w=None, 
                 lr_w=0.01, n_epochs=50):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        
        if initial_w is None:
            initial_w = np.array([[-8.0], [1.0]])
        self.initial_w = initial_w.copy()
        
        self.lr_w = lr_w
        self.n_epochs = n_epochs
        
        self.w_history = None
        self.w_final = None
        
        self.training_in_progress = False
        self.current_epoch = 0
        self.callback = None
    
    def train(self, callback=None):
        """Шаблонный метод, определяющий общую структуру алгоритма обучения"""
        self.callback = callback
        self.training_in_progress = True
        self.reset_epoch_counter()
        
        w = self.initial_w.copy()
        w_history = [w.flatten()]
        
        # Инициализация дополнительных параметров
        params = self.initialize_params()
        
        # Подготовка данных для обучения
        data_iterator = self.prepare_data_iterator()
        
        for epoch in range(self.n_epochs):
            self.current_epoch = epoch + 1
            
            # Выполнение одной эпохи обучения
            w, w_history, params = self.perform_epoch(w, w_history, params, data_iterator)
            
            # Обратный вызов для обновления UI
            if callback and epoch % 1 == 0:
                self.update_callback()
        
        # Сохранение результатов обучения
        self.save_training_results(w, np.array(w_history))
        self.training_in_progress = False
        
        return w, np.array(w_history)
    
    @abstractmethod
    def reset_epoch_counter(self):
        """Сбрасывает счетчик эпох"""
        pass
    
    @abstractmethod
    def initialize_params(self):
        """Инициализирует дополнительные параметры, если необходимо"""
        pass
    
    @abstractmethod
    def prepare_data_iterator(self):
        """Подготавливает итератор по данным"""
        pass
    
    @abstractmethod
    def perform_epoch(self, w, w_history, params, data_iterator):
        """Выполняет одну эпоху обучения"""
        pass
    
    @abstractmethod
    def update_callback(self):
        """Обновляет UI через callback"""
        pass
    
    @abstractmethod
    def save_training_results(self, w, w_history):
        """Сохраняет результаты обучения"""
        pass


class SGDTrainerNoBN(AbstractTrainer):
    """Реализация обучения без Batch Normalization"""
    
    def reset_epoch_counter(self):
        """Сбрасывает счетчик эпох"""
        self.current_epoch = 0
    
    def initialize_params(self):
        """Для обучения без BN не требуются дополнительные параметры"""
        return None
    
    def prepare_data_iterator(self):
        """Для обучения без BN используется весь датасет сразу"""
        return [(self.X, self.y)]
    
    def perform_epoch(self, w, w_history, params, data_iterator):
        """Выполняет одну эпоху обучения без BN"""
        X_full, y_full = data_iterator[0]
        
        grad_w = compute_batch_gradient_no_bn(X_full, y_full, w)
        w = w - self.lr_w * grad_w
        w_history.append(w.flatten())
        
        return w, w_history, params
    
    def update_callback(self):
        """Обновляет UI через callback"""
        if self.callback:
            self.callback(update_no_bn=True)
    
    def save_training_results(self, w, w_history):
        """Сохраняет результаты обучения"""
        self.w_final = w
        self.w_history = w_history


class SGDTrainerWithBN(AbstractTrainer):
    """Реализация обучения с Batch Normalization"""
    
    def __init__(self, X, y, batch_size, initial_w=None, 
                 lr_w=0.01, lr_bn=0.0001, n_epochs=50, epsilon_bn=1e-5, 
                 freeze_bn_params=False):
        super().__init__(X, y, batch_size, initial_w, lr_w, n_epochs)
        self.lr_bn = lr_bn
        self.epsilon_bn = epsilon_bn
        self.freeze_bn_params = freeze_bn_params
    
    def reset_epoch_counter(self):
        """Сбрасывает счетчик эпох"""
        self.current_epoch = 0
    
    def initialize_params(self):
        """Инициализирует параметры Batch Normalization"""
        n_features = self.X.shape[1]
        gamma = np.ones(n_features)
        beta = np.zeros(n_features)
        return {'gamma': gamma, 'beta': beta}
    
    def prepare_data_iterator(self):
        """Подготавливает итератор по мини-батчам данных"""
        return DataLoader(self.X, self.y, self.batch_size, shuffle=True)
    
    def perform_epoch(self, w, w_history, params, data_iterator):
        """Выполняет одну эпоху обучения с BN"""
        gamma = params['gamma']
        beta = params['beta']
        
        for X_batch, y_batch in data_iterator:
            if len(y_batch) == 0:
                continue
            
            grad_w, grad_gamma, grad_beta, _ = compute_bn_forward_and_grad(
                X_batch, y_batch, w, gamma, beta, self.epsilon_bn, compute_grads=True
            )
            
            if not self.freeze_bn_params:
                gamma = gamma - self.lr_bn * grad_gamma
                beta = beta - self.lr_bn * grad_beta
            
            w = w - self.lr_w * grad_w
            w_history.append(w.flatten())
        
        params['gamma'] = gamma
        params['beta'] = beta
        
        return w, w_history, params
    
    def update_callback(self):
        """Обновляет UI через callback"""
        if self.callback:
            self.callback(update_bn=True)
    
    def save_training_results(self, w, w_history):
        """Сохраняет результаты обучения"""
        self.w_final = w
        self.w_history = w_history


class SGDTrainer:
    """Класс для обучения моделей с обратной связью для GUI"""
    def __init__(self, X, y, batch_size, initial_w=None, 
                 lr_w=0.01, lr_bn=0.0001, n_epochs=50, epsilon_bn=1e-5):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        
        if initial_w is None:
            initial_w = np.array([[-8.0], [1.0]])
        self.initial_w = initial_w.copy()
        
        self.lr_w = lr_w
        self.lr_bn = lr_bn
        self.n_epochs = n_epochs
        self.epsilon_bn = epsilon_bn
        
        # Создаем экземпляры тренеров с разными стратегиями
        self.trainer_no_bn = SGDTrainerNoBN(X, y, batch_size, initial_w, lr_w, n_epochs)
        self.trainer_with_bn = SGDTrainerWithBN(X, y, batch_size, initial_w, lr_w, lr_bn, 
                                                n_epochs, epsilon_bn)
        
        # Для обратной совместимости
        self.w_hist_no_bn = None
        self.w_hist_bn = None
        self.w_final_no_bn = None
        self.w_final_bn = None
        
        self.training_in_progress = False
        self.epoch_no_bn = 0
        self.epoch_bn = 0
        self.callback = None
        
    def train_no_bn(self, callback=None):
        """Запуск обучения без Batch Normalization на всем наборе данных"""
        self.callback = callback
        self.training_in_progress = True
        
        # Делегируем обучение специализированному тренеру
        w, w_history = self.trainer_no_bn.train(callback)
        
        # Сохраняем результаты для обратной совместимости
        self.w_final_no_bn = w
        self.w_hist_no_bn = w_history
        self.epoch_no_bn = self.trainer_no_bn.current_epoch
        self.training_in_progress = False
        
        return w, w_history
    
    def train_with_bn(self, callback=None, freeze_bn_params=False):
        """Запуск обучения с Batch Normalization"""
        self.callback = callback
        self.training_in_progress = True
        
        # Обновляем параметр freeze_bn_params
        self.trainer_with_bn.freeze_bn_params = freeze_bn_params
        
        # Делегируем обучение специализированному тренеру
        w, w_history = self.trainer_with_bn.train(callback)
        
        # Сохраняем результаты для обратной совместимости
        self.w_final_bn = w
        self.w_hist_bn = w_history
        self.epoch_bn = self.trainer_with_bn.current_epoch
        self.training_in_progress = False
        
        return w, w_history
    
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