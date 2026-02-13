import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import smart_solver


def pause():
    """暂停等待用户按回车"""
    input("\n>>> 按回车继续...\n")


def format_array_inline(arr, name="", max_display=20, show_full=False):
    """格式化数组为单行显示"""
    if show_full or len(arr) <= max_display:
        return f"{name}[{', '.join(f'{x:.6g}' for x in arr)}]"
    else:
        head = ', '.join(f'{x:.6g}' for x in arr[:10])
        tail = ', '.join(f'{x:.6g}' for x in arr[-10:])
        return f"{name}[{head} ... {tail}] (共{len(arr)}个)"


def format_matrix_inline(A, name="A", max_rows=10, show_full=False):
    """格式化矩阵为多行显示，每行不换行"""
    is_sparse = sp.issparse(A)
    if is_sparse:
        A = A.toarray()
    
    m, n = A.shape
    lines = []
    
    if show_full or m <= max_rows:
        for i in range(m):
            row_str = ', '.join(f'{x:8.4g}' for x in A[i])
            lines.append(f"  [{row_str}]")
    else:
        # 显示前5行
        for i in range(5):
            row_str = ', '.join(f'{x:8.4g}' for x in A[i])
            lines.append(f"  [{row_str}]")
        lines.append(f"  ... (省略 {m-10} 行) ...")
        # 显示后5行
        for i in range(m-5, m):
            row_str = ', '.join(f'{x:8.4g}' for x in A[i])
            lines.append(f"  [{row_str}]")
    
    return f"{name} ({m}x{n}):\n" + '\n'.join(lines)


def print_matrix_info(A, b, x_true=None, auto_mode=False):
    """打印输入矩阵和向量信息"""
    is_sparse = sp.issparse(A)
    
    print("\n" + "="*80)
    print("【输入数据】")
    print("="*80)
    
    m, n = A.shape
    should_ask = (m * n > 1000)
    
    # 只询问一次（自动模式下不询问）
    show_full = False
    if should_ask and not auto_mode:
        choice = input(f"\n数据规模较大 (A: {m}x{n}, 总元素: {m*n})，是否展示完整数据？(y/n，默认n): ").strip().lower()
        show_full = (choice == 'y')
    
    # === 打印 A ===
    if is_sparse:
        print(f"\nA: 稀疏矩阵 {m}x{n}, 非零元素: {A.nnz}, 密度: {A.nnz/(m*n):.2%}")
    else:
        print(f"\nA: 稠密矩阵 {m}x{n}")
    
    if show_full or m * n <= 1000:
        print(format_matrix_inline(A, "A", max_rows=m, show_full=True))
    else:
        print(format_matrix_inline(A, "A", max_rows=10, show_full=False))
    
    # === 打印 b ===
    print(f"\nb: 向量长度 {len(b)}")
    if show_full or len(b) <= 100:
        print(format_array_inline(b, "b", max_display=len(b), show_full=True))
    else:
        print(format_array_inline(b, "b", max_display=20, show_full=False))
    
    # === 打印 x_true ===
    if x_true is not None:
        print(f"\nx_true: 真实解长度 {len(x_true)}")
        if show_full or len(x_true) <= 100:
            print(format_array_inline(x_true, "x_true", max_display=len(x_true), show_full=True))
        else:
            print(format_array_inline(x_true, "x_true", max_display=20, show_full=False))
    
    # 返回用户选择，供后续使用
    return show_full


def print_solution(x, A, b, name="求解结果", show_full=False):
    """打印求解结果"""
    is_sparse = sp.issparse(A)
    m, n = A.shape
    
    print(f"\n【{name}】")
    print("-"*80)
    
    # === 打印 x ===
    print(f"x: 解向量长度 {len(x)}")
    if show_full or len(x) <= 100:
        print(format_array_inline(x, "x", max_display=len(x), show_full=True))
    else:
        print(format_array_inline(x, "x", max_display=20, show_full=False))
    
    # === 打印 Ax ===
    Ax = A @ x
    print(f"\nA @ x: 结果向量长度 {len(Ax)}")
    if show_full or len(Ax) <= 100:
        print(format_array_inline(Ax, "A@x", max_display=len(Ax), show_full=True))
    else:
        print(format_array_inline(Ax, "A@x", max_display=20, show_full=False))
    
    # === 打印 b 对比 ===
    print(f"\nb (原始): 向量长度 {len(b)}")
    if show_full or len(b) <= 100:
        print(format_array_inline(b, "b", max_display=len(b), show_full=True))
    else:
        print(format_array_inline(b, "b", max_display=20, show_full=False))
    
    # === 打印残差 ===
    residual = b - Ax
    print(f"\n残差 (b - A@x): 向量长度 {len(residual)}")
    if show_full or len(residual) <= 100:
        print(format_array_inline(residual, "residual", max_display=len(residual), show_full=True))
    else:
        print(format_array_inline(residual, "residual", max_display=20, show_full=False))


def evaluate(case_name, A, b, x_true, x_est, verbose=True):
    """评价并打印结果，返回评价指标"""
    is_sparse = sp.issparse(A)
    
    # 计算残差
    residual = b - A @ x_est
    
    # 1. 相对残差
    rel_res = np.linalg.norm(residual) / np.linalg.norm(b) if np.linalg.norm(b) != 0 else np.linalg.norm(residual)
    
    # 2. 残差白噪性检测 (Durbin-Watson 统计量)
    if len(residual) > 1:
        diff_res = np.diff(residual)
        dw = np.sum(diff_res**2) / (np.sum(residual**2) + 1e-16)
    else:
        dw = float('nan')
        
    # 3. 参数相对误差
    rel_param_err = np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true) if np.linalg.norm(x_true) != 0 else np.linalg.norm(x_est)

    # 判断等级
    if rel_param_err < 1e-6:
        level = "excellent"  # 优秀
    elif rel_param_err < 1e-1:
        level = "good"  # 良好
    else:
        level = "warning"  # 警告

    if verbose:
        print(f"\n{'='*80}")
        print(f"【评价结果: {case_name}】")
        print(f"{'='*80}")
        print(f"1. 相对残差 (Relative Residual):        {rel_res:.2e}")
        print(f"2. 残差白噪性 (DW Statistic, ~2=white): {dw:.4f}")
        print(f"3. 参数相对误差 (Relative Param Error):  {rel_param_err:.2e}")
        
        if level == "warning":
            print("⚠️  警告: 求解结果与真值偏差较大！")
        elif level == "excellent":
            print("✅ 优秀: 求解精度极高！")
        else:
            print("✓  良好: 求解精度可接受")
    
    return {
        "rel_res": rel_res,
        "dw": dw,
        "rel_param_err": rel_param_err,
        "level": level
    }


def generate_case(choice, custom_m=None, custom_n=None):
    """生成测试用例"""
    m, n = custom_m or 100, custom_n or 10
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
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        S = np.logspace(0, -10, len(S)) 
        A = U @ np.diag(S) @ Vh
    elif choice == 6:
        name = "欠定问题 (m<n)"
        m, n = custom_m or 10, custom_n or 20
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
        b[0:5] += 100.0
        return name, A, b, x_true
    elif choice == 10:
        name = "过定大样本 (m>>n)"
        m, n = custom_m or 10000, custom_n or 5
        A = np.random.randn(m, n)
        x_true = np.random.randn(n)
    elif choice == 11:
        name = "病态 + 异常点 (The Toxic Combo)"
        A[:, 1] = A[:, 0] + np.random.randn(m) * 1e-12
        b = A @ x_true + np.random.randn(m) * noise_level
        b[0] += 50.0
        return name, A, b, x_true
    elif choice == 12:
        name = "大规模稀疏系统 (Large Sparse, N=10000)"
        N = custom_n or 10000
        main_diag = np.ones(N) * 2.0
        off_diag = np.ones(N-1) * -1.0
        A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csc')
        x_true = np.random.randn(N)
        b = A @ x_true
        return name, A, b, x_true

    b = A @ x_true + np.random.randn(m) * noise_level
    return name, A, b, x_true


def input_custom_data():
    """用户输入自定义数据"""
    print("\n" + "="*80)
    print("【自定义数据输入】")
    print("="*80)
    
    try:
        m = int(input("请输入行数 m: ").strip())
        n = int(input("请输入列数 n: ").strip())
        
        print("\n请选择输入方式:")
        print("1. 随机生成")
        print("2. 手动输入 A 和 b")
        choice = input("选择 (1/2): ").strip()
        
        if choice == '1':
            A = np.random.randn(m, n)
            x_true = np.random.randn(n)
            b = A @ x_true + 0.01 * np.random.randn(m)
            return "自定义随机数据", A, b, x_true
        
        elif choice == '2':
            print(f"\n请输入 A ({m}x{n})，每行用空格分隔:")
            A = []
            for i in range(m):
                row = list(map(float, input(f"第 {i+1} 行: ").strip().split()))
                if len(row) != n:
                    print(f"错误: 需要 {n} 个数，但输入了 {len(row)} 个")
                    return None
                A.append(row)
            A = np.array(A)
            
            print(f"\n请输入 b ({m})，用空格分隔:")
            b = np.array(list(map(float, input().strip().split())))
            if len(b) != m:
                print(f"错误: 需要 {m} 个数，但输入了 {len(b)} 个")
                return None
            
            # 尝试求真实解
            try:
                x_true = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                x_true = np.zeros(n)
            
            return "自定义手动输入", A, b, x_true
        
    except Exception as e:
        print(f"输入错误: {e}")
        return None


def run_test(choice, custom_m=None, custom_n=None, auto_mode=False):
    """运行单个测试"""
    np.random.seed(42 + choice)
    
    if choice == 99:  # 自定义数据
        result = input_custom_data()
        if result is None:
            return None
        name, A, b, x_true = result
    else:
        name, A, b, x_true = generate_case(choice, custom_m, custom_n)
    
    if not auto_mode:
        print(f"\n\n{'#'*80}")
        print(f"# 测试用例: {name}")
        print(f"{'#'*80}")
        
        # 打印输入信息，并获取用户的展示选择
        show_full = print_matrix_info(A, b, x_true, auto_mode)
    else:
        print(f"\n运行测试 {choice}: {name}...", end=" ")
        show_full = False
    
    is_sparse = sp.issparse(A)
    
    # ========== Baseline 求解器 ==========
    if not is_sparse:
        if not auto_mode:
            print("\n" + "="*80)
            print("【Baseline: Numpy lstsq】")
            print("="*80)
        x_baseline, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        if not auto_mode:
            print_solution(x_baseline, A, b, "Numpy lstsq 求解结果", show_full)
        baseline_result = evaluate(name + " (Numpy lstsq)", A, b, x_true, x_baseline, verbose=not auto_mode)
        baseline_name = "Numpy lstsq"
    else:
        if not auto_mode:
            print("\n" + "="*80)
            print("【Baseline: Scipy LSQR】")
            print("="*80)
        x_baseline = spla.lsqr(A, b)[0]
        if not auto_mode:
            print_solution(x_baseline, A, b, "Scipy LSQR 求解结果", show_full)
        baseline_result = evaluate(name + " (Scipy LSQR)", A, b, x_true, x_baseline, verbose=not auto_mode)
        baseline_name = "Scipy LSQR"
    
    # ========== SmartLinearSolver ==========
    if not auto_mode:
        print("\n" + "="*80)
        print("【SmartLinearSolver】")
        print("="*80)
    x_smart = smart_solver.solve(A, b, verbose=not auto_mode)
    if not auto_mode:
        print_solution(x_smart, A, b, "SmartLinearSolver 求解结果", show_full)
    smart_result = evaluate(name + " (SmartLinearSolver)", A, b, x_true, x_smart, verbose=not auto_mode)
    
    if auto_mode:
        print("完成")
    else:
        pause()
    
    return {
        "name": name,
        "baseline_name": baseline_name,
        "baseline": baseline_result,
        "smart": smart_result
    }


def print_comparison_table(results):
    """打印对比表格"""
    print("\n\n" + "="*100)
    print("【全测对比表格】")
    print("="*100)
    
    # 表头
    print(f"\n{'测试用例':<35} | {'Baseline 误差':<15} | {'Smart 误差':<15} | {'对比':<15}")
    print("-" * 100)
    
    # 统计
    baseline_better = 0
    smart_better = 0
    tie = 0
    
    for i, result in enumerate(results, 1):
        name = result['name']
        baseline_err = result['baseline']['rel_param_err']
        smart_err = result['smart']['rel_param_err']
        baseline_name = result['baseline_name']
        
        # 格式化误差
        baseline_str = f"{baseline_err:.2e}"
        smart_str = f"{smart_err:.2e}"
        
        # 判断谁更好
        if smart_err < baseline_err * 0.9:  # Smart 明显更好（至少好10%）
            comparison = "✓ Smart 更优"
            smart_better += 1
        elif baseline_err < smart_err * 0.9:  # Baseline 明显更好
            comparison = "✗ Baseline 更优"
            baseline_better += 1
        else:  # 相当
            comparison = "≈ 相当"
            tie += 1
        
        # 打印行
        print(f"{i:2d}. {name:<32} | {baseline_str:<15} | {smart_str:<15} | {comparison:<15}")
    
    print("="*100)
    
    # 统计总结
    total = len(results)
    print(f"\n【统计总结】")
    print(f"  SmartSolver 更优: {smart_better}/{total} ({smart_better/total*100:.1f}%)")
    print(f"  Baseline 更优:    {baseline_better}/{total} ({baseline_better/total*100:.1f}%)")
    print(f"  性能相当:         {tie}/{total} ({tie/total*100:.1f}%)")
    
    # 详细等级统计
    print(f"\n【详细等级统计】")
    
    baseline_excellent = sum(1 for r in results if r['baseline']['level'] == 'excellent')
    baseline_good = sum(1 for r in results if r['baseline']['level'] == 'good')
    baseline_warning = sum(1 for r in results if r['baseline']['level'] == 'warning')
    
    smart_excellent = sum(1 for r in results if r['smart']['level'] == 'excellent')
    smart_good = sum(1 for r in results if r['smart']['level'] == 'good')
    smart_warning = sum(1 for r in results if r['smart']['level'] == 'warning')
    
    print(f"  Baseline:     优秀 {baseline_excellent}, 良好 {baseline_good}, 警告 {baseline_warning}")
    print(f"  SmartSolver:  优秀 {smart_excellent}, 良好 {smart_good}, 警告 {smart_warning}")
    
    print("="*100)


def main():
    """主函数"""
    while True:
        print("\n" + "="*80)
        print("最小二乘求解测试框架")
        print("="*80)
        print("1.  良好条件 / 基准")
        print("2.  列尺度差异巨大")
        print("3.  强共线")
        print("4.  秩亏（重复列）")
        print("5.  近秩亏 / 病态")
        print("6.  欠定问题 (m<n)")
        print("7.  强噪声")
        print("8.  弱噪声")
        print("9.  含异常点 (outlier)")
        print("10. 过定大样本 (m>>n)")
        print("11. 病态 + 异常点 (The Toxic Combo)")
        print("12. 大规模稀疏系统 (Large Sparse, N=10000)")
        print("-"*80)
        print("0.  运行全部测试 (带暂停)")
        print("a.  全测模式 (不暂停 + 对比表格)")
        print("c.  自定义 m, n (使用预设测试)")
        print("d.  自定义数据 (手动输入)")
        print("q.  退出")
        print("="*80)
        
        choice = input("\n请选择: ").strip().lower()
        
        if choice == 'q':
            print("\n再见！")
            break
        
        elif choice == 'a':
            # 全测模式
            print("\n开始全测模式...")
            results = []
            for i in range(1, 13):
                result = run_test(i, auto_mode=True)
                if result:
                    results.append(result)
            print_comparison_table(results)
            pause()
        
        elif choice == 'd':
            run_test(99)
        
        elif choice == 'c':
            try:
                m = int(input("请输入 m (行数): ").strip())
                n = int(input("请输入 n (列数): ").strip())
                test_choice = int(input("请选择测试类型 (1-12): ").strip())
                if 1 <= test_choice <= 12:
                    run_test(test_choice, m, n)
                else:
                    print("❌ 无效的测试类型")
            except ValueError:
                print("❌ 请输入有效的数字")
        
        else:
            try:
                c = int(choice)
                if c == 0:
                    for i in range(1, 13):
                        run_test(i)
                    print("\n" + "="*80)
                    print("全部测试完成！")
                    print("="*80)
                elif 1 <= c <= 12:
                    run_test(c)
                else:
                    print("❌ 无效选择")
            except ValueError:
                print("❌ 请输入有效的选项")


if __name__ == "__main__":
    main()
