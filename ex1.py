import pennylane as qml
from pennylane import numpy as np
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory
from pennylane.noise import mitigate_with_zne
import matplotlib.pyplot as plt

# ------------------------------ 中文字体设置 ------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示异常

# ============================================
# 实验配置
# ============================================
n_wires = 4
n_layers = 1
template = qml.SimplifiedTwoDesign
np.random.seed(1967)

# 生成固定权重（保证所有噪声条件下电路一致）
weights_shape = template.shape(n_layers, n_wires)
w1, w2 = [2 * np.pi * np.random.random(s) for s in weights_shape]

# 待测试的噪声模型及其对应的通道类（使用 PennyLane 内置类）
noise_classes = {
    "相位阻尼": qml.PhaseDamping,
    "比特翻转": qml.BitFlip,
    "去极化": qml.DepolarizingChannel,
    "振幅阻尼": qml.AmplitudeDamping,
}

# 噪声参数列表
noise_strengths = [0.01, 0.03, 0.05, 0.08, 0.1]
scale_factors = [1, 2, 3]

# 存储结果
results = {name: {"strengths": [], "efficiency": []} for name in noise_classes}

# ============================================
# 定义电路（镜像电路 U·U†）
# ============================================
def circuit(w1, w2):
    template(w1, w2, wires=range(n_wires))
    qml.adjoint(template)(w1, w2, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))

# 理想设备（无噪声）
dev_ideal = qml.device("default.mixed", wires=n_wires)
ideal_qnode = qml.QNode(circuit, dev_ideal)
ideal_result = ideal_qnode(w1, w2)
print(f"理想值（无噪声）: {ideal_result:.6f}\n")

# ============================================
# 定义辅助函数：创建 NoiseModel
# ============================================
def get_noise_model(channel_class, p):
    """根据通道类和强度返回 NoiseModel"""
    fcond = qml.noise.wires_in(range(n_wires))
    noise_op = qml.noise.partial_wires(channel_class, p)
    return qml.NoiseModel({fcond: noise_op})

# ============================================
# 主循环：扫描噪声类型与强度
# ============================================
for noise_name, channel_cls in noise_classes.items():
    print(f"===== 正在测试 {noise_name} =====")
    for p in noise_strengths:
        noise_model = get_noise_model(channel_cls, p)

        dev_clean = qml.device("default.mixed", wires=n_wires)
        dev_noisy = qml.add_noise(dev_clean, noise_model=noise_model)
        noisy_qnode = qml.QNode(circuit, dev_noisy)
        # 分解为原生门（Mitiq 要求）
        noisy_qnode = qml.transforms.decompose(noisy_qnode, gate_set=["RY", "CZ"])

        # 未缓解的噪声结果
        noisy_result = noisy_qnode(w1, w2)

        # 应用 ZNE 缓解
        extrapolate = RichardsonFactory.extrapolate
        mitigated_qnode = mitigate_with_zne(
            noisy_qnode, scale_factors, fold_global, extrapolate
        )
        mitigated_result = mitigated_qnode(w1, w2)

        # 计算恢复效率（限制在 0~100）
        denominator = ideal_result - noisy_result
        if denominator != 0:
            efficiency = (mitigated_result - noisy_result) / denominator * 100
            efficiency = max(0.0, min(100.0, efficiency))
        else:
            efficiency = 100.0

        results[noise_name]["strengths"].append(p)
        results[noise_name]["efficiency"].append(efficiency)

        print(f"  p={p:.3f}: 噪声结果={noisy_result:.4f}, 缓解后={mitigated_result:.4f}, 缓解效率={efficiency:.1f}%")
    print()

# ============================================
# 绘制折线图（中文标签，保存为 SVG）
# ============================================
plt.figure(figsize=(9, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# 噪声名称已为中文，直接遍历
for (name, data), color, marker in zip(results.items(), colors, markers):
    plt.plot(data["strengths"], data["efficiency"],
             marker=marker, color=color, linewidth=2, markersize=8, label=name)

plt.xlabel("噪声参数 $p$", fontsize=12)
plt.ylabel("缓解效率 (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("zne_efficiency_comparison.svg", format="svg", dpi=300)
plt.show()

# 打印汇总表格（中文表头）
print("\n" + "=" * 70)
print("恢复效率汇总表 (%)")
print("=" * 70)
print(f"{'噪声类型':<12}", end="")
for p in noise_strengths:
    print(f"p={p:<5}", end="")
print()
for name, data in results.items():
    print(f"{name:<12}", end="")
    for eff in data["efficiency"]:
        print(f"{eff:<6.1f}", end="")
    print()