# compare_solvers_v8_mild.py
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, random
from read_g2o import *
from utility import *
from pose_graph import *
from plot_data import *
import smart_solver

DEBUG_MODE = True

def poison_data_mild(all_edges):
    edges_list = []
    for frame in all_edges:
        for edge in all_edges[frame]:
            edges_list.append(edge)
    n_edges = len(edges_list)
    print(f">> 正在注入【轻微】干扰: 总边数 {n_edges}")
    poisoned_indices = random.sample(range(n_edges), int(n_edges * 0.15))
    for idx in poisoned_indices:
        m = np.array(edges_list[idx]["measurement"])
        m[:2] *= 1.5 
        edges_list[idx]["measurement"] = m.tolist()
        edges_list[idx]["info_mat"] *= 2.0 
        edges_list[idx]["is_poisoned"] = True
    print(f">> 干扰完成: 注入 {len(poisoned_indices)} 个轻微噪声点。")
    return edges_list

def compute_huber_weight(e, threshold=1.0):
    if e <= threshold: return 1.0
    else: return threshold / e

def run_pgo_dual(filename, max_itr=5):
    all_vertex_ref, _, _, _ = parse_g2o_file(filename)
    x_gt = np.array([all_vertex_ref[v] for v in all_vertex_ref])
    all_vertex, all_edges, _, _ = parse_g2o_file(filename)
    edges_flat = poison_data_mild(all_edges)
    x_init = x_gt.flatten().reshape(-1, 1)
    n_states = x_init.shape[0]
    x_np, x_ss = np.copy(x_init), np.copy(x_init)
    
    def build_system(current_x, use_robust_kernel=False):
        H, b = np.zeros((n_states, n_states)), np.zeros((n_states,1))
        total_chi2, outlier_count = 0, 0
        for edge in edges_flat:
            i, j = int(edge["frame_1"]), int(edge["frame_2"])
            xi, xj = current_x[3*i:3*i+3, 0], current_x[3*j:3*j+3, 0]
            zij, info_mat = np.array(edge["measurement"]), np.array(edge["info_mat"])
            eij = cal_eij(xi, xj, zij)
            chi2 = eij.T @ info_mat @ eij
            error_norm = np.sqrt(chi2[0,0])
            if DEBUG_MODE and edge.get("is_poisoned", False) and itr==0 and not use_robust_kernel:
                if random.random() < 0.05:
                    print(f"   [DEBUG] 轻微干扰边 {i}->{j} | 误差 Norm: {error_norm:.2f} (预期中等)")
            weight = 1.0
            if use_robust_kernel:
                weight = compute_huber_weight(error_norm, threshold=1.5)
                if weight < 0.8: outlier_count += 1 
            robust_info = info_mat * weight
            total_chi2 += (eij.T @ robust_info @ eij)[0,0]
            Ai, Bi = cal_jac_A_B(xi, xj, zij)
            H[3*i:3*i+3, 3*i:3*i+3] += Ai.T @ robust_info @ Ai
            H[3*i:3*i+3, 3*j:3*j+3] += Ai.T @ robust_info @ Bi
            H[3*j:3*j+3, 3*i:3*i+3] += Bi.T @ robust_info @ Ai
            H[3*j:3*j+3, 3*j:3*j+3] += Bi.T @ robust_info @ Bi
            b[3*i:3*i+3, 0] += np.squeeze(Ai.T @ robust_info @ eij)
            b[3*j:3*j+3, 0] += np.squeeze(Bi.T @ robust_info @ eij)
        H[0:3, 0:3] += np.eye(3)
        return H, b, total_chi2, outlier_count

    print("\n[开始对抗实验] 正在处理轻微干扰数据...")
    for itr in range(max_itr):
        H_np, b_np, err_np, _ = build_system(x_np, use_robust_kernel=False)
        x_np += np.linalg.inv(H_np) @ -b_np
        H_ss, b_ss, err_ss, n_out = build_system(x_ss, use_robust_kernel=True)
        x_ss += smart_solver.solve(H_ss, -b_ss, verbose=False).reshape(-1, 1)
        print(f"迭代 {itr}: Numpy误差={err_np:.2e} | Smart误差={err_ss:.2e} (抑制 {n_out} 个点)")

    rmsd_np = np.sqrt(np.mean(np.linalg.norm(x_np.reshape(-1,3)[:,:2] - x_gt[:,:2], axis=1)**2))
    rmsd_ss = np.sqrt(np.mean(np.linalg.norm(x_ss.reshape(-1,3)[:,:2] - x_gt[:,:2], axis=1)**2))

    print("\n" + "="*80)
    print("                对抗实验总结: 轻微干扰 (Mild Noise)")
    print("="*80)
    print(f"{'指标':<20} | {'Numpy (受干扰)':<20} | {'SmartSolver (精细化)'}")
    print("-" * 80)
    print(f"{'平均漂移 (RMSD)':<20} | {rmsd_np:<20.4f} m | {rmsd_ss:.4f} m")
    print(f"{'精度提升倍数':<20} | {'Baseline':<20} | {rmsd_np/(rmsd_ss+1e-6):.1f}x")
    print("-" * 80)
    
    plt.figure(figsize=(15, 10))
    plt.plot(x_gt[:, 0], x_gt[:, 1], color='black', linestyle='--', label='Ground Truth', alpha=0.4)
    plt.plot(x_np.reshape(-1,3)[:, 0], x_np.reshape(-1,3)[:, 1], color='red', label=f'Numpy (Drift: {rmsd_np:.1f}m)', alpha=0.7)
    plt.plot(x_ss.reshape(-1,3)[:, 0], x_ss.reshape(-1,3)[:, 1], color='#32CD32', label=f'SmartSolver (Precise: {rmsd_ss:.1f}m)', linewidth=1.5)
    plt.title("Comparison on Mildly Noisy Data")
    plt.legend()
    plt.axis('equal')
    os.makedirs('output_results', exist_ok=True)
    plt.savefig("output_results/experiment_mild_intel.png", dpi=200)
    print(f"\n[成功] 对比图已生成: output_results/experiment_mild_intel.png")

if __name__ == "__main__":
    run_pgo_dual('data/intel.g2o', max_itr=5)
