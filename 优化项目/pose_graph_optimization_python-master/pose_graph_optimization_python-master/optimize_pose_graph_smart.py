# optimize_pose_graph_smart.py
from read_g2o import *
from utility import *
from pose_graph import *
from plot_data import *
import numpy as np
import math
import sys
import argparse
import smart_solver
import matplotlib.pyplot as plt

def run_comparison_optimization(filename, fix_node=0, max_itr=10):
    # read the g2o file
    all_vertex, all_edges, anchor_frame, dim = parse_g2o_file(filename)

    # 保存初始状态图
    draw_2d_all_states(all_vertex, all_edges, draw_start_end_node=1, plot_immediate=0, save_filename="initial_v1_4.png")

    # 准备状态向量
    x_smart = []
    for ref_frame in all_vertex:
        x_smart.append(all_vertex[ref_frame])
    x_smart = np.array(x_smart).reshape(-1,1)
    
    # 克隆一个用于对比的 x (Numpy 方案)
    x_numpy = np.copy(x_smart)
    
    n_states = x_smart.shape[0]
    print(f'位姿节点数量: {n_states/3}')
    print("-" * 50)

    for itr in range(max_itr):
        H = np.zeros((n_states, n_states))
        b = np.zeros((n_states,1))
        total_error = 0

        # 构建 Hessian 和梯度 (这里逻辑两套方法公用同一套 H, b 构造，只是解法不同)
        for frame in all_edges:
            for edge in all_edges[frame]:
                i, j = int(edge["frame_1"]), int(edge["frame_2"])
                x_i = x_smart[3*i:3*i+3, 0]
                x_j = x_smart[3*j:3*j+3, 0]
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

        # 固定节点
        H[fix_node*3:(fix_node+1)*3, fix_node*3:(fix_node+1)*3] += np.eye(3)

        print(f"\n>>> 迭代轮次: {itr} | 当前总误差: {total_error[0,0]:.4f}")
        
        # 1. Numpy 解法 (用于展示可能存在的问题)
        try:
            # 原代码用 np.linalg.inv，在大规模或病态时极不稳定
            delta_x_numpy = np.linalg.inv(H) @ -b
            numpy_status = "OK"
        except Exception as e:
            delta_x_numpy = np.zeros_like(b)
            numpy_status = f"FAILED ({str(e)})"

        # 2. SmartLinearSolver 解法
        print(f"[SmartLinearSolver 介入进行诊断与求解...]")
        delta_x_smart = smart_solver.solve(H, -b, verbose=True)
        
        # 打印对比数据
        diff_norm = np.linalg.norm(delta_x_smart - delta_x_numpy)
        print(f"解的差异范数 (Smart vs Numpy): {diff_norm:.2e}")
        
        # 更新状态 (采用 Smart 解进行后续迭代)
        x_smart += delta_x_smart.reshape(-1, 1)

        # 保存每一轮的轨迹
        if itr % 2 == 0 or itr == max_itr - 1:
            temp_vertex = create_vertex_from_state_vector(x_smart.reshape(-1, 3))
            draw_2d_all_states(temp_vertex, all_edges, draw_start_end_node=1, plot_immediate=0, save_filename=f'itr_{itr}_smart.png')

    print("\n" + "="*50)
    print("优化完成！最终轨迹已保存为图片。")
    print("="*50)

    # 最终可视化
    final_vertex = create_vertex_from_state_vector(x_smart.reshape(-1, 3))
    # 这里我们强制保存一张最终大图
    plt.figure(figsize=(10, 8))
    for frame in final_vertex:
        pose = final_vertex[frame]
        plt.plot(pose[0], pose[1], 'bo')
        plt.text(pose[0], pose[1], str(frame))
    plt.title("Final Optimized Trajectory (Smart Solver)")
    plt.savefig("final_optimized_smart.png")
    print("已保存最终结果图: final_optimized_smart.png")

if __name__ == "__main__":
    filename = 'test_datasetv2.g2o'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    run_comparison_optimization(filename)
