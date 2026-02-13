import numpy as np
import math
from utility import *

def save_g2o(filename, poses, edges):
    with open(filename, 'w') as f:
        for i, p in enumerate(poses):
            f.write(f"VERTEX_SE2 {i} {p[0]} {p[1]} {p[2]}\n")
        for e in edges:
            info = "100.0 0.0 0.0 100.0 0.0 100.0"
            f.write(f"EDGE_SE2 {e[0]} {e[1]} {e[2]} {e[3]} {e[4]} {info}\n")

def generate_complex_outlier_case():
    """场景 A: 极端离群点。构造一个大环路，但闭环约束极其离谱。"""
    n_nodes = 40
    poses_gt = []
    # 构造一个 10x10 的矩形环路
    for i in range(10): poses_gt.append([float(i), 0.0, 0.0])
    for i in range(10): poses_gt.append([10.0, float(i), np.pi/2])
    for i in range(10): poses_gt.append([10.0-i, 10.0, np.pi])
    for i in range(10): poses_gt.append([0.0, 10.0-i, -np.pi/2])
    
    poses_gt = np.array(poses_gt)
    edges = []
    # 里程计约束 (带点小噪声)
    for i in range(len(poses_gt)-1):
        d = poses_gt[i+1] - poses_gt[i]
        edges.append([i, i+1, d[0], d[1], d[2]])
    
    # 加入正常的闭环 (39 -> 0)
    edges.append([39, 0, 1.0, 0.0, -np.pi/2])
    
    # 加入一个“致命”的假闭环 (Outlier): 节点 5 和 节点 25 
    # 它们本该相距很远，但我强行说它们在同一位置，且权重很高
    edges.append([5, 25, 0.0, 0.0, 0.0])
    
    # 为了让初始值别太离谱，我们给 VERTEX 稍微加点位移噪声
    poses_noisy = poses_gt + np.random.normal(0, 0.5, poses_gt.shape)
    
    save_g2o('case_outlier.g2o', poses_noisy, edges)
    np.save('case_outlier_gt.npy', poses_gt)
    print("场景 A 生成完成: 包含一个严重错误的闭环约束。")

def generate_weak_constraint_case():
    """场景 B: 零空间。节点 5 和 6 之间完全断开。"""
    n_nodes = 15
    poses_gt = []
    for i in range(n_nodes): poses_gt.append([float(i), 0.0, 0.0])
    poses_gt = np.array(poses_gt)
    
    edges = []
    for i in range(n_nodes-1):
        if i == 7: continue # 故意掐断 7 和 8 之间的联系
        edges.append([i, i+1, 1.0, 0.0, 0.0])
        
    # 给初始值加噪声，否则优化器不动
    poses_noisy = poses_gt + np.random.normal(0, 0.2, poses_gt.shape)
    save_g2o('case_weak.g2o', poses_noisy, edges)
    np.save('case_weak_gt.npy', poses_gt)
    print("场景 B 生成完成: 包含一个断裂的动力学链条 (Hessian 奇异)。")

if __name__ == "__main__":
    generate_complex_outlier_case()
    generate_weak_constraint_case()
