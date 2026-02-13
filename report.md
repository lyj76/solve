# 1.直接调用numpy的结果
=== 最小二乘求解测试框架 ===
1. 良好条件 / 基准
2. 列尺度差异巨大
3. 强共线
4. 秩亏（重复列）
5. 近秩亏 / 病态
6. 欠定问题 (m<n)
7. 强噪声
8. 弱噪声
9. 含异常点 (outlier)
10. 过定大样本 (m>>n)
11. 病态 + 异常点 (The Toxic Combo)
0. 运行全部
q. 退出

请选择例子编号: 0

--- 评价结果: 良好条件 / 基准 ---
1. 相对残差 (Relative Residual): 3.99e-07
2. 残差白噪性 (DW Statistic, ~2 is white): 1.9129
3. 参数相对误差 (Relative Parameter Error): 1.80e-07

--- 评价结果: 列尺度差异巨大 ---
1. 相对残差 (Relative Residual): 2.16e-14
2. 残差白噪性 (DW Statistic, ~2 is white): 2.0667
3. 参数相对误差 (Relative Parameter Error): 2.95e-01
警告: 求解结果与真值偏差较大！

--- 评价结果: 强共线 ---
1. 相对残差 (Relative Residual): 2.67e-07
2. 残差白噪性 (DW Statistic, ~2 is white): 1.8543
3. 参数相对误差 (Relative Parameter Error): 9.81e+00
警告: 求解结果与真值偏差较大！

--- 评价结果: 秩亏（重复列） ---
1. 相对残差 (Relative Residual): 2.80e-07
2. 残差白噪性 (DW Statistic, ~2 is white): 1.9985
3. 参数相对误差 (Relative Parameter Error): 2.19e-01
警告: 求解结果与真值偏差较大！

--- 评价结果: 近秩亏 / 病态 ---
1. 相对残差 (Relative Residual): 1.74e-04
2. 残差白噪性 (DW Statistic, ~2 is white): 1.9733
3. 参数相对误差 (Relative Parameter Error): 1.61e+03
警告: 求解结果与真值偏差较大！

--- 评价结果: 欠定问题 (m<n) ---
1. 相对残差 (Relative Residual): 1.12e-15
2. 残差白噪性 (DW Statistic, ~2 is white): 0.8782
3. 参数相对误差 (Relative Parameter Error): 5.01e-01
警告: 求解结果与真值偏差较大！

--- 评价结果: 强噪声 ---
1. 相对残差 (Relative Residual): 1.88e-01
2. 残差白噪性 (DW Statistic, ~2 is white): 1.6197
3. 参数相对误差 (Relative Parameter Error): 8.39e-02

--- 评价结果: 弱噪声 ---
1. 相对残差 (Relative Residual): 1.75e-11
2. 残差白噪性 (DW Statistic, ~2 is white): 1.7557
3. 参数相对误差 (Relative Parameter Error): 7.18e-12

--- 评价结果: 含异常点 (outlier) ---
1. 相对残差 (Relative Residual): 9.59e-01
2. 残差白噪性 (DW Statistic, ~2 is white): 0.3381
3. 参数相对误差 (Relative Parameter Error): 2.68e+00
警告: 求解结果与真值偏差较大！

--- 评价结果: 过定大样本 (m>>n) ---
1. 相对残差 (Relative Residual): 4.75e-07
2. 残差白噪性 (DW Statistic, ~2 is white): 2.0092
3. 参数相对误差 (Relative Parameter Error): 1.15e-08

--- 评价结果: 病态 + 异常点 (The Toxic Combo) ---
1. 相对残差 (Relative Residual): 8.43e-01
2. 残差白噪性 (DW Statistic, ~2 is white): 1.0597
3. 参数相对误差 (Relative Parameter Error): 1.34e+11
警告: 求解结果与真值偏差较大！