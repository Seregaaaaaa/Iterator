import pygame
import pygame_gui
import sys
import threading
import pygame.gfxdraw
import numpy as np
import io
from PIL import Image

from src.data.data_generator import generate_data
from src.models.trainer import SGDTrainer
from src.visualization.plot_utils import plot_contour_paths

# Константы для приложения
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
BG_COLOR = (240, 240, 240)
TEXT_COLOR = (10, 10, 10)
FPS = 60

class BatchNormVisualizationApp:
    """Основной класс приложения"""
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Визуализация обучения с Batch Normalization и без")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Инициализация pygame_gui
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Создаем данные
        self.X, self.y, self.true_w = generate_data(n_samples=200, std_dev=1.0)
        
        # Создаем тренер
        self.trainer = SGDTrainer(
            self.X, self.y, batch_size=64, 
            initial_w=np.array([[-8.0], [1.0]]),
            lr_w=0.01, lr_bn=0.0001, n_epochs=50
        )
        
        # Сохраняем поверхности для рисования
        self.contour_surface = None
        self.contour_rect = None
        
        # Настройка UI
        self.setup_ui()
        
        # Флаги для процессов
        self.no_bn_training_thread = None
        self.bn_training_thread = None
        
    def setup_ui(self):
        # Заголовок и пояснение
        self.title_rect = pygame.Rect(50, 10, SCREEN_WIDTH - 100, 40)
        
        # Настройки панель
        self.control_panel = pygame.Rect(20, 50, 240, SCREEN_HEIGHT - 70)
        
        # Параметры слева
        self.batch_size_label_rect = pygame.Rect(30, 70, 100, 20)
        self.batch_size_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(30, 95, 200, 20),
            start_value=64, value_range=(4, 128), manager=self.manager
        )
        self.batch_size_value_rect = pygame.Rect(140, 70, 60, 20)
        
        self.epochs_label_rect = pygame.Rect(30, 125, 100, 20)
        self.epochs_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(30, 150, 200, 20),
            start_value=50, value_range=(10, 100), manager=self.manager
        )
        self.epochs_value_rect = pygame.Rect(140, 125, 60, 20)
        
        self.lr_w_label_rect = pygame.Rect(30, 180, 100, 20)
        self.lr_w_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(30, 205, 200, 20),
            start_value=0.01, value_range=(0.001, 0.1), manager=self.manager
        )
        self.lr_w_value_rect = pygame.Rect(140, 180, 60, 20)
        
        self.lr_bn_label_rect = pygame.Rect(30, 235, 100, 20)
        self.lr_bn_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(30, 260, 200, 20),
            start_value=0.0001, value_range=(0.00001, 0.01), manager=self.manager
        )
        self.lr_bn_value_rect = pygame.Rect(140, 235, 60, 20)
        
        # Кнопки для запуска
        self.train_no_bn_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(30, 320, 200, 50),
            text="Обучить без BN", manager=self.manager
        )
        
        self.train_bn_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(30, 380, 200, 50),
            text="Обучить с BN", manager=self.manager
        )
        
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(30, 440, 200, 50),
            text="Сбросить", manager=self.manager
        )
        
        # Метки результатов справа внизу
        self.results_panel = pygame.Rect(SCREEN_WIDTH - 500, SCREEN_HEIGHT - 150, 480, 130)
        self.progress_label_rect = pygame.Rect(SCREEN_WIDTH - 490, SCREEN_HEIGHT - 140, 460, 20)
        self.no_bn_weights_rect = pygame.Rect(SCREEN_WIDTH - 490, SCREEN_HEIGHT - 110, 460, 20)
        self.bn_weights_rect = pygame.Rect(SCREEN_WIDTH - 490, SCREEN_HEIGHT - 80, 460, 20)
        self.true_weights_rect = pygame.Rect(SCREEN_WIDTH - 490, SCREEN_HEIGHT - 50, 460, 20)
    
    def update_gui(self, update_no_bn=False, update_bn=False):
        """Обновляет GUI по мере обучения"""
        if update_no_bn:
            pass  # Просто будем перерисовывать в основном цикле
        if update_bn:
            pass  # Просто будем перерисовывать в основном цикле
    
    def update_plot(self):
        """Обновляет визуализацию с текущими весами"""
        histories = []
        labels = []
        
        # Добавляем историю без BN, если она есть
        if self.trainer.w_hist_no_bn is not None:
            histories.append(self.trainer.w_hist_no_bn)
            labels.append('Без BN')
        
        # Добавляем историю с BN, если она есть
        if self.trainer.w_hist_bn is not None:
            histories.append(self.trainer.w_hist_bn)
            labels.append('С BN')
        
        # Если есть истории, создаем визуализацию
        if histories:
            title = "Сравнение SGD с BN и без BN"
            raw_data, size = plot_contour_paths(
                self.X, self.y, histories, labels,
                self.trainer.initial_w, self.true_w, title,
                fig_size=(8, 6), dpi=100  # Увеличиваем размер и DPI графика
            )
            
            surf = pygame.image.fromstring(raw_data, size, "RGB")
            self.contour_surface = surf
            # Увеличиваем размер области для графика и смещаем немного влево
            self.contour_rect = surf.get_rect(center=(SCREEN_WIDTH//2 + 80, SCREEN_HEIGHT//2 - 50))
    
    def draw(self):
        """Отрисовка всех элементов"""
        self.screen.fill(BG_COLOR)
        
        # Заголовок
        font_title = pygame.font.SysFont('Arial', 28, bold=True)
        title_surf = font_title.render("Визуализация обучения с Batch Normalization и без", True, TEXT_COLOR)
        self.screen.blit(title_surf, (self.title_rect.centerx - title_surf.get_width()//2, self.title_rect.top))
        
        # Панель управления
        pygame.draw.rect(self.screen, (220, 220, 220), self.control_panel, border_radius=5)
        
        # Параметры
        font_params = pygame.font.SysFont('Arial', 14)  # Уменьшаем размер шрифта для текста над слайдерами
        batch_label = font_params.render("Размер батча:", True, TEXT_COLOR)
        batch_value = font_params.render(f"{int(self.batch_size_slider.get_current_value())}", True, TEXT_COLOR)
        self.screen.blit(batch_label, self.batch_size_label_rect)
        self.screen.blit(batch_value, self.batch_size_value_rect)
        
        epochs_label = font_params.render("Число эпох:", True, TEXT_COLOR)
        epochs_value = font_params.render(f"{int(self.epochs_slider.get_current_value())}", True, TEXT_COLOR)
        self.screen.blit(epochs_label, self.epochs_label_rect)
        self.screen.blit(epochs_value, self.epochs_value_rect)
        
        lr_w_label = font_params.render("Learning rate w:", True, TEXT_COLOR)
        lr_w_value = font_params.render(f"{self.lr_w_slider.get_current_value():.4f}", True, TEXT_COLOR)
        self.screen.blit(lr_w_label, self.lr_w_label_rect)
        self.screen.blit(lr_w_value, self.lr_w_value_rect)
        
        lr_bn_label = font_params.render("Learning rate BN:", True, TEXT_COLOR)
        lr_bn_value = font_params.render(f"{self.lr_bn_slider.get_current_value():.6f}", True, TEXT_COLOR)
        self.screen.blit(lr_bn_label, self.lr_bn_label_rect)
        self.screen.blit(lr_bn_value, self.lr_bn_value_rect)
        
        # Контур визуализации
        if self.contour_surface:
            self.screen.blit(self.contour_surface, self.contour_rect)
        
        # Панель результатов
        pygame.draw.rect(self.screen, (230, 230, 230), self.results_panel, border_radius=5)
        
        # Прогресс обучения
        progress_text = "Прогресс: "
        if self.trainer.training_in_progress:
            if self.no_bn_training_thread and self.no_bn_training_thread.is_alive():
                progress_text += f"обучение без BN (эпоха {self.trainer.epoch_no_bn}/{int(self.epochs_slider.get_current_value())})"
            elif self.bn_training_thread and self.bn_training_thread.is_alive():
                progress_text += f"обучение с BN (эпоха {self.trainer.epoch_bn}/{int(self.epochs_slider.get_current_value())})"
        else:
            progress_text += "готово"
        
        progress_surf = font_params.render(progress_text, True, TEXT_COLOR)
        self.screen.blit(progress_surf, self.progress_label_rect)
        
        # Веса
        weights = self.trainer.get_final_weights()
        
        if weights['no_bn'] is not None:
            no_bn_text = f"Веса без BN: w1={weights['no_bn'][0,0]:.4f}, w2={weights['no_bn'][1,0]:.4f}"
            no_bn_surf = font_params.render(no_bn_text, True, TEXT_COLOR)
            self.screen.blit(no_bn_surf, self.no_bn_weights_rect)
        
        if weights['with_bn'] is not None:
            bn_text = f"Веса с BN: w1={weights['with_bn'][0,0]:.4f}, w2={weights['with_bn'][1,0]:.4f}"
            bn_surf = font_params.render(bn_text, True, TEXT_COLOR)
            self.screen.blit(bn_surf, self.bn_weights_rect)
        
        true_text = f"Истинные веса: w1={weights['true_w'][0,0]:.4f}, w2={weights['true_w'][1,0]:.4f}"
        true_surf = font_params.render(true_text, True, TEXT_COLOR)
        self.screen.blit(true_surf, self.true_weights_rect)
        
        # Обновляем GUI элементы
        self.manager.draw_ui(self.screen)
        
    def reset(self):
        """Сбрасывает все обучение"""
        # Останавливаем потоки, если они запущены
        self.trainer.training_in_progress = False
        
        if self.no_bn_training_thread and self.no_bn_training_thread.is_alive():
            self.no_bn_training_thread.join(timeout=0.1)
        
        if self.bn_training_thread and self.bn_training_thread.is_alive():
            self.bn_training_thread.join(timeout=0.1)
        
        # Создаем новые данные и тренера
        self.X, self.y, self.true_w = generate_data(n_samples=200, std_dev=1.0)
        
        batch_size = int(self.batch_size_slider.get_current_value())
        lr_w = self.lr_w_slider.get_current_value()
        lr_bn = self.lr_bn_slider.get_current_value()
        n_epochs = int(self.epochs_slider.get_current_value())
        
        self.trainer = SGDTrainer(
            self.X, self.y, batch_size=batch_size,
            initial_w=np.array([[-8.0], [1.0]]),
            lr_w=lr_w, lr_bn=lr_bn, n_epochs=n_epochs
        )
        
        # Сбрасываем визуализацию
        self.contour_surface = None
        self.contour_rect = None
    
    def run(self):
        """Основной цикл приложения"""
        running = True
        while running:
            time_delta = self.clock.tick(FPS) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.USEREVENT:
                    if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == self.train_no_bn_button:
                            # Запуск обучения без BN в отдельном потоке
                            if not self.trainer.training_in_progress:
                                # Обновляем параметры
                                self.trainer.batch_size = int(self.batch_size_slider.get_current_value())
                                self.trainer.lr_w = self.lr_w_slider.get_current_value()
                                self.trainer.n_epochs = int(self.epochs_slider.get_current_value())
                                
                                self.no_bn_training_thread = threading.Thread(
                                    target=self.trainer.train_no_bn,
                                    args=(self.update_gui,)
                                )
                                self.no_bn_training_thread.daemon = True
                                self.no_bn_training_thread.start()
                        
                        elif event.ui_element == self.train_bn_button:
                            # Запуск обучения с BN в отдельном потоке
                            if not self.trainer.training_in_progress:
                                # Обновляем параметры
                                self.trainer.batch_size = int(self.batch_size_slider.get_current_value())
                                self.trainer.lr_w = self.lr_w_slider.get_current_value()
                                self.trainer.lr_bn = self.lr_bn_slider.get_current_value()
                                self.trainer.n_epochs = int(self.epochs_slider.get_current_value())
                                
                                self.bn_training_thread = threading.Thread(
                                    target=self.trainer.train_with_bn,
                                    args=(self.update_gui,)
                                )
                                self.bn_training_thread.daemon = True
                                self.bn_training_thread.start()
                        
                        elif event.ui_element == self.reset_button:
                            self.reset()
                
                self.manager.process_events(event)
            
            # Обновляем GUI
            self.manager.update(time_delta)
            
            # Обновляем визуализацию
            self.update_plot()
            
            # Отрисовка
            self.draw()
            
            pygame.display.flip()
        
        pygame.quit()