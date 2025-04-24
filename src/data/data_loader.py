import numpy as np
from abc import ABC, abstractmethod

class Iterator(ABC):
    """Абстрактный интерфейс для итератора"""
    @abstractmethod
    def __iter__(self):
        pass
    
    @abstractmethod
    def __next__(self):
        pass
    
    @abstractmethod
    def has_next(self):
        pass

class Aggregate(ABC):
    """Абстрактный интерфейс для агрегата (коллекции)"""
    @abstractmethod
    def create_iterator(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

class DataIterator(Iterator):
    """Конкретная реализация итератора для данных"""
    def __init__(self, data_aggregate):
        self.data_aggregate = data_aggregate
        self.current_batch = 0
        self.indices = np.arange(self.data_aggregate.n_samples)
        if self.data_aggregate.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.has_next():
            raise StopIteration
        
        start = self.current_batch * self.data_aggregate.batch_size
        end = min(start + self.data_aggregate.batch_size, self.data_aggregate.n_samples)
        idx = self.indices[start:end]
        
        self.current_batch += 1
        return self.data_aggregate.X[idx], self.data_aggregate.y[idx]
    
    def has_next(self):
        return self.current_batch < self.data_aggregate.n_batches

class DataAggregate(Aggregate):
    """Конкретная реализация агрегата (коллекции данных)"""
    def __init__(self, X, y, batch_size, shuffle=True):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
    
    def create_iterator(self):
        return DataIterator(self)
    
    def __len__(self):
        return self.n_batches

# Для обратной совместимости с существующим кодом
class DataLoader:
    """Класс-обертка для обратной совместимости"""
    def __init__(self, X, y, batch_size, shuffle=True):
        self.aggregate = DataAggregate(X, y, batch_size, shuffle)
        
    def __iter__(self):
        self.iterator = self.aggregate.create_iterator()
        return self.iterator
    
    def __next__(self):
        return next(self.iterator)
    
    def __len__(self):
        return len(self.aggregate)