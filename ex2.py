import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import json
import time
import warnings

# 尝试导入 PennyLane
try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError:
    raise ImportError("请安装依赖: pip install pennylane scikit-learn matplotlib")

warnings.filterwarnings("ignore")


# ==================== 1. 环境与中文字体设置 ====================
def setup_chinese_font():
    import matplotlib.font_manager as fm
    candidates = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    for font in candidates:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[字体] 使用中文字体: {font}")
            return font
    return None


CHINESE_FONT = setup_chinese_font()


# ==================== 2. 数据加载与预处理 ====================
def load_fashion_mnist_binary(n_samples=100, classes=(0, 1), random_state=1359):
    from sklearn.datasets import fetch_openml
    print(f"\n 正在加载 Fashion-MNIST 数据集 (类别: {classes})...")
    data = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X, y = data.data.astype(np.float32) / 255.0, data.target.astype(np.int64)

    mask = np.isin(y, classes)
    X_sub, y_sub = X[mask], y[mask]
    y_binary = np.where(y_sub == classes[0], 1, -1)

    np.random.seed(random_state)
    idx_pos = np.where(y_binary == 1)[0]
    idx_neg = np.where(y_binary == -1)[0]
    n_each = n_samples // 2
    idx = np.concatenate([np.random.choice(idx_pos, n_each), np.random.choice(idx_neg, n_each)])
    np.random.shuffle(idx)

    return X_sub[idx], y_binary[idx]


# ==================== 3. IQP 核函数与 ZNE 逻辑 ====================
def iqp_feature_map(features, wires, n_repeats=2):
    qml.IQPEmbedding(features=features, wires=wires, n_repeats=n_repeats)


def create_kernels(n_wires, n_repeats, noise_strength):
    wires = list(range(n_wires))
    dev = qml.device("default.mixed", wires=n_wires)

    def apply_noise(p):
        for w in wires: qml.DepolarizingChannel(p, wires=w)

    @qml.qnode(dev)
    def q_ideal(x1, x2):
        iqp_feature_map(x1, wires, n_repeats)
        qml.adjoint(iqp_feature_map)(x2, wires, n_repeats)
        return qml.probs(wires=wires)

    @qml.qnode(dev)
    def q_noisy(x1, x2):
        iqp_feature_map(x1, wires, n_repeats)
        apply_noise(noise_strength)
        qml.adjoint(iqp_feature_map)(x2, wires, n_repeats)
        apply_noise(noise_strength)
        return qml.probs(wires=wires)

    # ZNE: 5点外推
    scale_factors = [1.0, 1.2, 1.4, 1.6, 1.8]

    def q_zne(x1, x2):
        vals = []
        for s in scale_factors:
            @qml.qnode(dev)
            def _q(x1_v, x2_v, s_val):
                iqp_feature_map(x1_v, wires, n_repeats)
                apply_noise(noise_strength * s_val)
                qml.adjoint(iqp_feature_map)(x2_v, wires, n_repeats)
                apply_noise(noise_strength * s_val)
                return qml.probs(wires=wires)

            vals.append(_q(x1, x2, s)[0])

        if not hasattr(q_zne, "_first_call_done"):
            print(f"\n[ZNE诊断] 第一对样本各尺度核值: {np.round(vals, 6)}")
            q_zne._first_call_done = True

        coeffs = np.polyfit(scale_factors, vals, deg=2)
        return np.clip(coeffs[2], 0.0, 1.0)

    return (lambda x1, x2: float(q_ideal(x1, x2)[0])), \
        (lambda x1, x2: float(q_noisy(x1, x2)[0])), \
        (lambda x1, x2: float(q_zne(x1, x2)))


# ==================== 4. 矩阵计算（核心：带进度的统一接口） ====================
def compute_matrix_with_progress(X1, X2, kernel_func, desc="矩阵"):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    is_symmetric = (np.array_equal(X1, X2))

    print(f"  [{desc}] {n1}x{n2}")
    start_time = time.time()

    for i in range(n1):
        start_j = i if is_symmetric else 0
        for j in range(start_j, n2):
            val = kernel_func(X1[i], X2[j])
            K[i, j] = val
            if is_symmetric:
                K[j, i] = val

        if (i + 1) % 10 == 0 or (i + 1) == n1:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (n1 - i - 1)
            print(f"    进度: {i + 1}/{n1} | 已用: {elapsed:.1f}s | 预计剩余: {eta:.1f}s")

    return K


# ==================== 5. 评估指标 ====================
def evaluate_metrics(K, Y):
    # 核-目标对齐度 (Centered Kernel Alignment)
    n = len(Y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    T = np.outer(Y, Y)
    Tc = H @ T @ H
    align = np.sum(Kc * Tc) / (np.linalg.norm(Kc) * np.linalg.norm(Tc) + 1e-9)
    # 类间分离度
    same_mask = np.outer(Y, Y) > 0
    sep = np.mean(K[same_mask]) - np.mean(K[~same_mask])
    return float(align), float(sep)


def make_psd(K):
    eigvals, eigvecs = np.linalg.eigh((K + K.T) / 2)
    return eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T


# ==================== 6. 可视化布局（中文标题、SVG 输出） ====================
def plot_full_comparison(results, K_matrices, config):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 中文名称映射
    names_cn = ['理想', '噪声', 'ZNE缓解']
    metric_keys = ['alignment', 'separation', 'accuracy']
    metric_labels_cn = ['核目标对齐度', '类间分离度', 'SVM 测试准确率']
    colors = ['#2196F3', '#FF5722', '#4CAF50']

    # 上行：核矩阵热图
    for i, name_cn in enumerate(names_cn):
        ax = axes[0, i]
        im = ax.imshow(K_matrices[name_cn], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"{name_cn}核矩阵（训练集）")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 下行：核心指标柱状图
    for i, (m_key, m_label_cn) in enumerate(zip(metric_keys, metric_labels_cn)):
        ax = axes[1, i]
        vals = [results[name_cn][m_key] for name_cn in names_cn]
        bars = ax.bar(names_cn, vals,
                      color=colors, alpha=0.8, edgecolor='black', width=0.6)
        ax.set_title(m_label_cn, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    # 移除总标题（原 suptitle 已注释）
    # plt.suptitle(
    #     f"ZNE 实验分析（量子比特: {config['n_wires']}, 样本数: {config['n_samples']}, 噪声强度: {config['noise_strength']}）",
    #     fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存为 SVG 格式
    save_path = f"results_n{config['n_samples']}_w{config['n_wires']}_r{config['n_repeats']}_noise{config['noise_strength']}.svg"
    plt.savefig(save_path, format="svg", dpi=200)
    print(f"\n 结果图表已保存为: {save_path}")


# ==================== 7. 主程序循环 ====================
def main():
    # --- 用户配置区 ---
    config = {
        'n_samples': 100,
        'test_ratio': 0.3,
        'n_wires': 4,  # 比特数
        'n_repeats': 4,  # IQP层数
        'noise_strength': 0.1,  # 噪声强度
        'classes': (2, 4),
        'seed': 1359
    }

    print("=" * 70)
    print("IQP量子核 + ZNE 错误缓解")
    print("=" * 70)

    # 1. 数据准备
    X_raw, Y = load_fashion_mnist_binary(config['n_samples'], config['classes'], config['seed'])
    X_pca = PCA(n_components=config['n_wires']).fit_transform(X_raw)
    # 关键：缩放到 pi/4 以增强大规模比特下的稳定性
    X_scaled = MinMaxScaler(feature_range=(0, np.pi / 4)).fit_transform(X_pca)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=config['test_ratio'], random_state=config['seed'], stratify=Y
    )

    # 2. 核函数初始化
    k_fns = create_kernels(config['n_wires'], config['n_repeats'], config['noise_strength'])
    device_names = ['理想', '噪声', 'ZNE缓解']

    results = {}
    K_matrices = {}

    # 3. 核心评估循环
    for name, fn in zip(device_names, k_fns):
        print(f"\n>>> 正在评估设备: {name}")

        # 计算训练集 (对称阵)
        K_tr = compute_matrix_with_progress(X_train, X_train, fn, desc=f"{name}-训练集")
        K_matrices[name] = K_tr

        # 计算测试集 (非对称阵 - 新增进度条)
        K_te = compute_matrix_with_progress(X_test, X_train, fn, desc=f"{name}-测试集")

        # 计算物理指标
        align, sep = evaluate_metrics(K_tr, Y_train)

        # SVM 训练与预测
        svm = SVC(kernel='precomputed', C=10.0)
        svm.fit(make_psd(K_tr), Y_train)
        acc = float(np.mean(svm.predict(K_te) == Y_test))

        results[name] = {'alignment': align, 'separation': sep, 'accuracy': acc}

        print(f"  [结果] 准确率: {acc:.4f} | 对齐度: {align:.4f} | 分离度: {sep:.4f}")

    # 4. 最终表格输出（中文表头）
    print("\n" + "=" * 75)
    print(f"{'设备类型':<15} {'对齐度 (Align)':<18} {'分离度 (Sep)':<18} {'测试准确率 (Acc)':<15}")
    print("-" * 75)
    for n in device_names:
        r = results[n]
        print(f"{n:<15} {r['alignment']:<18.4f} {r['separation']:<18.4f} {r['accuracy']:<15.4f}")
    print("=" * 75)

    # 5. 可视化
    plot_full_comparison(results, K_matrices, config)


if __name__ == "__main__":
    main()