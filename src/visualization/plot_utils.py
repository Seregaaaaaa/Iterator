import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
from src.utils.losses import compute_full_loss_no_bn

def plot_contour_paths(X_full, y_full, histories, labels, initial_w, true_w, title, fig_size=(10, 7), dpi=100):
    """Создает визуализацию контуров потерь и траекторий обучения"""
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fig.add_subplot(111)
    n_paths = len(histories)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_paths))  # Разные цвета
    
    # Если нет траекторий, не рисуем контуры
    if n_paths == 0 or all(h is None for h in histories):
        ax.set_title("Нет данных для отображения", fontsize=9)
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        plt.close(fig)
        return raw_data, size
    
    # Prepare valid histories
    valid_histories = [h for h in histories if h is not None]
    if not valid_histories:
        ax.set_title("Нет данных для отображения", fontsize=9)
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        plt.close(fig)
        return raw_data, size
    
    # --- Контуры ---
    all_w1 = np.concatenate([h[:, 0] for h in valid_histories if h is not None] + [[initial_w[0,0]]])
    all_w2 = np.concatenate([h[:, 1] for h in valid_histories if h is not None] + [[initial_w[1,0]]])
    w1_min, w1_max = all_w1.min() - 1, all_w1.max() + 1
    w2_min, w2_max = all_w2.min() - 1, all_w2.max() + 1
    w1_vals = np.linspace(w1_min, w1_max, 70)
    w2_vals = np.linspace(w2_min, w2_max, 70)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    losses = np.zeros(W1.shape)
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w_ij = np.array([[W1[i, j]], [W2[i, j]]])
            losses[i, j] = compute_full_loss_no_bn(X_full, y_full, w_ij)
    
    levels = np.logspace(np.log10(losses.min()+1e-1), np.log10(losses.max()), 15)
    contour = ax.contour(W1, W2, losses, levels=levels, cmap='Greys', alpha=0.6)
    
    # --- Траектории ---
    valid_labels = [label for label, h in zip(labels, histories) if h is not None]
    valid_colors = colors[:len(valid_histories)]
    
    for i, (hist, label) in enumerate(zip(valid_histories, valid_labels)):
        ax.plot(hist[:, 0], hist[:, 1], '-', color=valid_colors[i], alpha=0.7, linewidth=2.5, label=f'Путь ({label})')
        ax.plot(hist[-1, 0], hist[-1, 1], 'o', color=valid_colors[i], markersize=10, label=f'Конец ({label})')
    
    # --- Общие точки ---
    ax.plot(initial_w[0,0], initial_w[1,0], 'go', markersize=12, label='Старт')  # Увеличен размер маркера
    if true_w is not None:
        ax.plot(true_w[0,0], true_w[1,0], 'm*', markersize=14, label='Истинный минимум')  # Увеличен размер маркера
    
    ax.set_xlabel("Вес w1", fontsize=12)  # Увеличен размер шрифта
    ax.set_ylabel("Вес w2", fontsize=12)  # Увеличен размер шрифта
    ax.set_title(title, fontsize=14)  # Увеличен размер заголовка
    ax.set_xlim(w1_min, w1_max)
    ax.set_ylim(w2_min, w2_max)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)  # Увеличен размер шрифта легенды
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=9)  # Увеличен размер чисел на осях
    
    # Convert to surface
    fig.tight_layout()
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    plt.close(fig)
    
    return raw_data, size