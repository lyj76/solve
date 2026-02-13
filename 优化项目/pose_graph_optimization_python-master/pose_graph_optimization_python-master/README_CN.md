# 智能位姿图优化 (Smart PGO) 项目说明文档

本目录包含一个集成 **SmartLinearSolver (SLS)** 的位姿图优化 (Pose Graph Optimization) 实验环境。

## 1. 项目结构
* `compare_solvers.py`: **核心运行脚本**。它会同时运行 Numpy 标准求解器和 SmartLinearSolver 求解器，并对比两者的收敛性能。
* `smart_solver.py`: 智能数值求解器（核心库）。
* `pose_graph.py` / `utility.py`: PGO 相关的位姿变换与雅可比计算逻辑。
* `output_results/`: **结果输出目录**。包含优化前后的轨迹对比图及误差下降曲线。

## 2. 核心数学改进
在 SLAM 的后端优化中，Hessian 矩阵常因以下原因变得病态：
1. **闭环不足**：导致零空间维度增加。
2. **起始点未固定**：导致整体平移漂移。
3. **数据异常点**：错误的闭环（False Loop Closure）会引入极大残差。

**SmartLinearSolver** 通过以下链路解决上述问题：
* **自动缩放 (Column Scaling)**: 消除旋转与平移量纲差异导致的病态。
* **岭回归 (Ridge Regression)**: 保证在矩阵奇异时仍能输出物理上合理的 delta 位姿。
* **鲁棒回归 (IRLS)**: 自动识别并抑制错误的闭环约束。

## 3. 运行方法
请在当前目录下执行：
```bash
MPLBACKEND=Agg python3 compare_solvers.py
```

## 4. 实验预期结果
运行结束后，您可以查看 `output_results/` 文件夹：
* `final_comparison.png`: 展示初始轨迹、Numpy 优化轨迹与 SmartSolver 优化轨迹的重合/差异情况。
* `error_convergence.png`: 展示两套方案在迭代过程中的残差下降速度。

## 5. 诊断报告说明
SmartSolver 在运行时会实时输出“体检报告”，您可以观察其 **Condition Number (条件数)** 和 **Reliability (可靠性评分)**，以判断当前优化步的数值稳定性。
