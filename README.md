# SmartLinearSolver (SLS) - 智能数值线性代数“宪兵”系统

**Project Title:** SmartLinearSolver - An Automated Diagnostic & Remediation Middleware for Ill-posed Linear Systems

## 1. 项目背景与痛点 (Background & Problem)

在机器人控制（逆运动学）、化学计量学（光谱分析）及多项式拟合等工程场景中，核心数学问题往往归结为 $Ax=b$。然而，现有的标准库（如 `numpy.linalg.solve`）存在严重的“静默失败”风险：

* **黑盒化：** 面对病态矩阵（Ill-conditioned Matrix），直接输出数值溢出的垃圾解，无任何警告。
* **缺乏解释：** 工程师无法得知计算失败是源于数据共线性、噪声干扰还是传感器故障。
* **规模瓶颈：** 对于 $N > 10000$ 的大规模稀疏问题，传统 $O(N^3)$ 的直接解法（Dense Solver）在内存和时间上均不可接受。

**本项目旨在构建一个“数值中间件”，通过数学诊断自动化，为 $Ax=b$ 提供可解释、高鲁棒且可扩展的求解方案。**

---

## 2. 核心功能架构 (Core Architecture)

本项目采用 **"Diagnosis-Evidence-Probe"** 三段式处理流，并在 V2 版本中引入了 **"Iterative Engine"** 以支持大规模计算。

### 模块一：全景诊断 (Diagnosis Engine)
* **静默介入：** 支持 `import smart_solver as ss` 后直接调用，接口与 Numpy 保持一致。
* **多维体检：**
    * **病态性：** 自动计算条件数 $\kappa(A)$ 和秩 (Rank)。
    * **结构检测：** 识别列尺度差异（Column Scaling）并自动归一化。
    * **异常点：** 基于 MAD 和残差分布自动识别 Outliers。

### 模块二：智能求解 (Smart Solving)
* **自适应策略路由：**
    * **良态小规模：** 自动路由至 QR/Cholesky 分解（极速）。
    * **病态/秩亏：** 自动路由至 **Ridge Regression (岭回归)**，利用 L-Curve/GCV 自动寻参。
    * **脏数据/异常：** 自动触发 **IRLS (迭代重加权最小二乘)**，动态剔除错误观测。
    * **大规模稀疏 (V2)：** 自动路由至 **LSQR/LSMR** 迭代求解器。

### 模块三：迭代正则化 (V2: Iterative Regularization)
* **时间换精度：** 对于超大规模问题，利用 Krylov 子空间投影进行求解。
* **早停即正则 (Early Stopping)：** 利用迭代次数作为正则化参数，防止过拟合噪声。
* **Anytime Solving：** 支持设定时间/内存预算，在资源受限时返回当前最优近似解。

---

## 3. 技术实现细节 (Implementation)

* **语言/栈：** Python 3.9+, NumPy, SciPy (Sparse)
* **核心算法：**
    * **SVD**：用于小规模核心诊断与岭回归参数搜索。
    * **Huber Kernel**：用于鲁棒回归的 M-Estimator。
    * **LSQR**：用于大规模稀疏系统的迭代求解。
* **数值保护：**
    * **Scaling Floor**：防止极小列范数导致的数值爆炸。
    * **Explosion Guard**：自动检测解范数异常并回退策略。

---

## 4. 测试与验证 (Validation)

项目包含 `test.py` 测试框架，覆盖 12 类典型场景：
1. 良好条件 / 基准
2. 列尺度差异巨大 (Scaling Test)
3. 强共线 (Collinearity)
4. 秩亏 (Rank Deficient)
5. 近秩亏 / 病态 (Ill-conditioned)
6. 欠定问题 ($m < n$)
7. 强噪声
8. 弱噪声
9. 含异常点 (Outlier)
10. 过定大样本 ($m \gg n$)
11. 病态 + 异常点 (The Toxic Combo)
12. **大规模稀疏系统 (Large Sparse, V2)**

---

## 5. 快速开始 (Quick Start)

### 基础求解
```python
import smart_solver as ss
# 自动处理缩放、病态和异常点
x = ss.solve(A, b, verbose=True)
```

### 鲁棒性对比实验
```bash
# 运行 Intel 数据集投毒测试 (验证 IRLS 能力)
python3 solve/优化项目/.../compare_solvers_v7.py
```

### 全量测试
```bash
# 运行所有 12 个测试用例
echo "0" | python3 solve/test.py
```
