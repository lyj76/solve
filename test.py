import numpy as np
import smart_solver

def evaluate(case_name, A, b, x_true, x_est):
    """评价并打印结果"""
    residual = b - A @ x_est
    # 1. 相对残差
    rel_res = np.linalg.norm(residual) / np.linalg.norm(b) if np.linalg.norm(b) != 0 else np.linalg.norm(residual)
    
    # 2. 残差白噪性检测 (使用 Durbin-Watson 统计量)
    # DW = sum((e_i - e_{i-1})^2) / sum(e_i^2)
    # 接近 2 表示无自相关（白噪声），接近 0 或 4 表示强相关
    if len(residual) > 1:
        diff_res = np.diff(residual)
        dw = np.sum(diff_res**2) / np.sum(residual**2)
    else:
        dw = float('nan')
        
    # 3. 参数相对误差 (因为是模拟数据，我们知道真值)
    rel_param_err = np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true) if np.linalg.norm(x_true) != 0 else np.linalg.norm(x_est)

    print(f"\n--- 评价结果: {case_name} ---")
    print(f"1. 相对残差 (Relative Residual): {rel_res:.2e}")
    print(f"2. 残差白噪性 (DW Statistic, ~2 is white): {dw:.4f}")
    print(f"3. 参数相对误差 (Relative Parameter Error): {rel_param_err:.2e}")
    if rel_param_err > 1e-1:
        print("警告: 求解结果与真值偏差较大！")

def generate_case(choice):
    m, n = 100, 10
    x_true = np.random.randn(n)
    A = np.random.randn(m, n)
    noise_level = 1e-6
    
    name = ""
    if choice == 1:
        name = "良好条件 / 基准"
    elif choice == 2:
        name = "列尺度差异巨大"
        A[:, 0] *= 1e8
        A[:, 1] *= 1e-8
    elif choice == 3:
        name = "强共线"
        A[:, 1] = A[:, 0] + np.random.randn(m) * 1e-9
    elif choice == 4:
        name = "秩亏（重复列）"
        A[:, 1] = A[:, 0]
    elif choice == 5:
        name = "近秩亏 / 病态"
        # 使用简单的病态构造：奇异值快速衰减
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        S = np.logspace(0, -10, len(S)) 
        A = U @ np.diag(S) @ Vh
    elif choice == 6:
        name = "欠定问题 (m<n)"
        m, n = 10, 20
        A = np.random.randn(m, n)
        x_true = np.random.randn(n)
    elif choice == 7:
        name = "强噪声"
        noise_level = 0.5
    elif choice == 8:
        name = "弱噪声"
        noise_level = 1e-10
    elif choice == 9:
        name = "含异常点 (outlier)"
        b = A @ x_true + np.random.randn(m) * noise_level
        b[0:5] += 100.0  # 加入强离群点
        return name, A, b, x_true
    elif choice == 10:
        name = "过定大样本 (m>>n)"
        m, n = 10000, 5
        A = np.random.randn(m, n)
        x_true = np.random.randn(n)
    elif choice == 11:
        name = "病态 + 异常点 (The Toxic Combo)"
        # 强共线
        A[:, 1] = A[:, 0] + np.random.randn(m) * 1e-12
        b = A @ x_true + np.random.randn(m) * noise_level
        b[0] += 50.0
        return name, A, b, x_true
    elif choice == 12:
        name = "大规模稀疏系统 (Large Sparse, N=10000)"
        import scipy.sparse as sp
        N = 10000
        # 构建一个三对角矩阵 (模拟 1D 拉普拉斯算子)
        main_diag = np.ones(N) * 2.0
        off_diag = np.ones(N-1) * -1.0
        A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csc')
        x_true = np.random.randn(N)
        b = A @ x_true
        # 稀疏矩阵不加噪声，主要测求解能力
        return name, A, b, x_true

    b = A @ x_true + np.random.randn(m) * noise_level
    return name, A, b, x_true

def run_test(choice):
    name, A, b, x_true = generate_case(choice)
    
    print(f"\n{'='*80}\n测试用例: {name}\n{'='*80}")
    
    # 1. 稠密矩阵跑 Numpy lstsq，稀疏矩阵跑 Scipy LSQR
    import scipy.sparse as sp
    is_sparse = sp.issparse(A)
    
    if not is_sparse:
        x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        print("\n[Numpy lstsq 结果]")
        evaluate(name + " (Numpy)", A, b, x_true, x_lstsq)
    else:
        import scipy.sparse.linalg as spla
        print("\n[Scipy LSQR 结果 (Baseline)]")
        x_lsqr = spla.lsqr(A, b)[0]
        evaluate(name + " (LSQR)", A, b, x_true, x_lsqr)
    
    # 2. 使用 SmartLinearSolver
    print("\n[SmartLinearSolver 诊断与结果]")
    x_smart = smart_solver.solve(A, b, verbose=True)
    evaluate(name + " (Smart)", A, b, x_true, x_smart)

def main():
    while True:
        print("\n=== 最小二乘求解测试框架 ===")
        print("1. 良好条件 / 基准")
        print("2. 列尺度差异巨大")
        print("3. 强共线")
        print("4. 秩亏（重复列）")
        print("5. 近秩亏 / 病态")
        print("6. 欠定问题 (m<n)")
        print("7. 强噪声")
        print("8. 弱噪声")
        print("9. 含异常点 (outlier)")
        print("10. 过定大样本 (m>>n)")
        print("11. 病态 + 异常点 (The Toxic Combo)")
        print("12. 大规模稀疏系统 (Large Sparse, N=10000)")
        print("0. 运行全部")
        print("q. 退出")
        
        choice = input("\n请选择例子编号: ")
        if choice.lower() == 'q':
            break
        try:
            c = int(choice)
            if c == 0:
                for i in range(1, 13):
                    run_test(i)
            elif 1 <= c <= 12:
                run_test(c)
            else:
                print("无效选择")
        except ValueError:
            print("请输入数字或 q")

if __name__ == "__main__":
    main()
