# compare_solvers.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from read_g2o import *
from utility import *
from pose_graph import *
from plot_data import *
import smart_solver

def run_optimization(filename, solver_type="numpy", fix_node=0, max_itr=5):
    all_vertex, all_edges, anchor_frame, dim = parse_g2o_file(filename)
    x = []
    for ref_frame in all_vertex:
        x.append(all_vertex[ref_frame])
    x = np.array(x).reshape(-1,1)
    n_states = x.shape[0]
    
    errors = []
    
    for itr in range(max_itr):
        H = np.zeros((n_states, n_states))
        b = np.zeros((n_states,1))
        total_error = 0
        for frame in all_edges:
            for edge in all_edges[frame]:
                i, j = int(edge["frame_1"]), int(edge["frame_2"])
                x_i, x_j = x[3*i:3*i+3, 0], x[3*j:3*j+3, 0]
                zij, info_mat = edge["measurement"], edge["info_mat"]
                eij = cal_eij(x_i, x_j, zij)
                total_error += eij.T @ info_mat @ eij
                Aij, Bij = cal_jac_A_B(x_i, x_j, zij)
                H[3*i:3*i+3, 3*i:3*i+3] += Aij.T @ info_mat @ Aij
                H[3*i:3*i+3, 3*j:3*j+3] += Aij.T @ info_mat @ Bij
                H[3*j:3*j+3, 3*i:3*i+3] += Bij.T @ info_mat @ Aij
                H[3*j:3*j+3, 3*j:3*j+3] += Bij.T @ info_mat @ Bij
                b[3*i:3*i+3, 0] += np.squeeze(Aij.T @ info_mat @ eij)
                b[3*j:3*j+3, 0] += np.squeeze(Bij.T @ info_mat @ eij)

        H[fix_node*3:(fix_node+1)*3, fix_node*3:(fix_node+1)*3] += np.eye(3)
        errors.append(total_error[0,0])

        if solver_type == "numpy":
            try:
                delta_x = np.linalg.inv(H) @ -b
            except:
                delta_x = np.zeros_like(b)
        else:
            # 只有 smart 模式下打印诊断报告
            print(f"\n[SmartSolver 诊断 - 第 {itr} 轮]")
            delta_x = smart_solver.solve(H, -b, verbose=True).reshape(-1, 1)
            
        x += delta_x
        
    return x, errors

def main():
    input_file = 'test_datasetv2.g2o'
    output_dir = 'output_results'
    
    print("="*60)
    print("      Pose Graph Optimization: Solver Comparison Experiment")
    print("="*60)
    
    # 获取初始位姿
    all_vertex, all_edges, _, _ = parse_g2o_file(input_file)
    x_init = []
    for ref_frame in all_vertex:
        x_init.append(all_vertex[ref_frame])
    x_init = np.array(x_init).reshape(-1, 3) # 包含 (x, y, theta)
    
    # 运行 Numpy 优化
    print("\n正在运行 Numpy 标准优化 (Baseline)...")
    x_numpy_final, errors_numpy = run_optimization(input_file, solver_type="numpy")
    
    # 运行 Smart 优化
    print("\n正在运行 SmartLinearSolver 优化 (Pro)...")
    x_smart_final, errors_smart = run_optimization(input_file, solver_type="smart")
    
    # 打印最终对比报告
    print("\n" + " "*20 + "对比报告 (Final Comparison)")
    print("-" * 60)
    print(f"{'指标 (Metric)':<20} | {'Numpy (Inv)':<15} | {'SmartSolver':<15}")
    print("-" * 60)
    print(f"{'初始误差 (Init Error)':<20} | {errors_numpy[0]:<15.4f} | {errors_smart[0]:<15.4f}")
    print(f"{'最终误差 (Final Error)':<20} | {errors_numpy[-1]:<15.4f} | {errors_smart[-1]:<15.4f}")
    print(f"{'误差下降率 (%)':<20} | {(1-errors_numpy[-1]/errors_numpy[0])*100:<15.2f} | {(1-errors_smart[-1]/errors_smart[0])*100:<15.2f}")
    print("-" * 60)

    # 绘制对比图
    plt.figure(figsize=(12, 8))
    
    # 初始轨迹
    x_i = np.array(x_init).reshape(-1, 3)
    plt.plot(x_i[:, 0], x_i[:, 1], 'r--', label='Initial (Noisy)', alpha=0.5)
    
    # Numpy 结果
    x_n = x_numpy_final.reshape(-1, 3)
    plt.plot(x_n[:, 0], x_n[:, 1], 'g-o', label='Numpy Optimized')
    
    # Smart 结果
    x_s = x_smart_final.reshape(-1, 3)
    plt.plot(x_s[:, 0], x_s[:, 1], 'b-s', label='SmartSolver Optimized', markersize=4)
    
    plt.legend()
    plt.title("PGO Results Comparison: Numpy vs SmartSolver")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    
    save_path = os.path.join(output_dir, "final_comparison.png")
    plt.savefig(save_path)
    print(f"\n对比图已保存至: {save_path}")
    
    # 保存误差曲线
    plt.figure()
    plt.plot(errors_numpy, 'g', label='Numpy Error')
    plt.plot(errors_smart, 'b--', label='Smart Error')
    plt.yscale('log')
    plt.title("Error Convergence Comparison")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "error_convergence.png"))
    
    print(f"误差下降曲线已保存至: {output_dir}/error_convergence.png")

if __name__ == "__main__":
    main()
