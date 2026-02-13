import numpy as np
from scipy import optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class SmartLinearSolver:
    def __init__(self, A, b):
        self.is_sparse = sp.issparse(A)
        self.b_raw = np.asarray(b, dtype=float).flatten()
        self.m = A.shape[0]
        self.n = A.shape[1]
        
        # 稀疏矩阵特殊处理
        if self.is_sparse:
            self.A_raw = A
            # 稀疏矩阵的列范数计算需要高效方法
            # 转换成 csc 格式可以快速访问列
            if not sp.isspmatrix_csc(A):
                self.A_csc = A.tocsc()
            else:
                self.A_csc = A
            
            # 计算列范数 (近似或精确)
            # 这里计算精确 L2 范数
            self.col_norms = np.sqrt(self.A_csc.power(2).sum(axis=0)).A1
        else:
            self.A_raw = np.asarray(A, dtype=float)
            self.col_norms = np.linalg.norm(self.A_raw, axis=0)
            
        # 2. 自动列缩放 (Column Scaling)
        med_norm = np.median(self.col_norms) if len(self.col_norms) > 0 else 1.0
        self.s_floor = max(1e-12, med_norm * 1e-10)
        self.col_norms = np.maximum(self.col_norms, self.s_floor)
        
        # 懒惰缩放：如果是稀疏矩阵，我们不显式构造 A = A_raw @ D^-1
        # 而是利用 LinearOperator
        if not self.is_sparse:
            self.A = self.A_raw / self.col_norms
        else:
            # 定义缩放算子
            D_inv = sp.diags(1.0 / self.col_norms)
            self.A = self.A_raw @ D_inv
            
        self.b = self.b_raw
        self.U, self.s, self.Vh = None, None, None
        self.u_dot_b = None
        self.alpha_final = 0
        self.diagnostics = {}
        self.weights = np.ones(self.m)
        self.advice = []
        self.x_scaled_norm = 0
        
    def _perform_svd(self, A_current=None):
        if self.is_sparse:
            return None, None, None, None # 稀疏模式不跑 SVD
            
        if A_current is None:
            if self.U is None:
                self.U, self.s, self.Vh = np.linalg.svd(self.A, full_matrices=False)
                self.u_dot_b = self.U.T @ self.b
            return self.U, self.s, self.Vh, self.u_dot_b
        else:
            U, s, Vh = np.linalg.svd(A_current, full_matrices=False)
            return U, s, Vh, U.T @ self.b

    def diagnose(self):
        self.diagnostics = {
            "shape": (self.m, self.n),
            "scale_stats": (np.min(self.col_norms), np.median(self.col_norms), np.max(self.col_norms))
        }
        
        if self.is_sparse:
            self.cond_raw = float('inf') # 稀疏大矩阵默认假设病态，不强求算 cond
            self.diagnostics['cond_raw'] = "N/A (Sparse)"
            self.diagnostics['cond_scaled'] = "N/A (Sparse)"
            self.diagnostics['rank'] = "N/A"
            self.strategy = "Iterative Solver (LSQR)"
            self.advice.append("检测到大规模稀疏矩阵，自动切换至迭代求解器。")
            return

        # 稠密矩阵逻辑
        U, s, Vh, ub = self._perform_svd()
        self.cond_raw = s[0] / s[-1] if s[-1] > 0 else float('inf')
        cond_scaled = s[0] / s[-1] if s[-1] > 0 else float('inf')
        tol = max(self.m, self.n) * np.finfo(float).eps * s[0]
        rank = np.sum(s > tol)
        
        self.diagnostics.update({
            "cond_raw": self.cond_raw,
            "cond_scaled": cond_scaled,
            "rank": rank
        })
        
        if self.m < self.n:
            self.strategy = "Minimum Norm"
            self.advice.append("欠定系统。")
        elif self.cond_raw > 1e8 or rank < self.n:
            self.strategy = "Ridge Regression (岭回归)"
            self.advice.append(f"原始系统病态 (Cond:{self.cond_raw:.1e})。")
        else:
            self.strategy = "Standard"

    def find_alpha_hybrid(self, U, s, Vh, ub):
        if self.m <= self.n: return 0
        def gcv_obj(a):
            f = s**2 / (s**2 + a**2)
            res_sq = np.sum(((1-f)*ub)**2) + (np.sum(self.b**2) - np.sum(ub**2))
            denom = (self.m - np.sum(f))**2
            return res_sq / denom if denom > 0 else np.inf
        a_gcv = optimize.minimize_scalar(gcv_obj, bounds=(1e-15, s[0]), method='bounded').x
        alphas = np.logspace(-15, np.log10(s[0]), 40)
        log_res, log_sol = [], []
        for a in alphas:
            f = s**2 / (s**2 + a**2)
            log_res.append(np.log(np.sqrt(np.sum(((1-f)*ub)**2)) + 1e-16))
            log_sol.append(np.log(np.sqrt(np.sum((f/s*ub)**2)) + 1e-16))
        eta, rho = np.array(log_sol), np.array(log_res)
        curv = np.abs(np.gradient(rho)*np.gradient(np.gradient(eta)) - np.gradient(np.gradient(rho))*np.gradient(eta))
        a_lcurve = alphas[np.argmax(curv)]
        return np.sqrt(a_gcv * a_lcurve)

    def solve(self):
        self.diagnose()
        
        # === 分支 1: 稀疏迭代求解 ===
        if self.is_sparse:
            # LSQR 自带正则化 (damp)
            # 我们可以简单设定一个小的 damp 作为基础正则化
            damp = 1e-6 
            # 求解 A_scaled * x_scaled = b
            res = spla.lsqr(self.A, self.b, damp=damp, iter_lim=self.n*2, show=False)
            x_s = res[0]
            self.alpha_final = damp
            x = x_s / self.col_norms
            self.advice.append(f"LSQR 迭代完成，停止原因: {res[1]} (Iter: {res[2]})")
            return x

        # === 分支 2: 稠密直接求解 ===
        U, s, Vh, ub = self._perform_svd()
        if "Ridge" in self.strategy:
            self.alpha_final = self.find_alpha_hybrid(U, s, Vh, ub)
        else:
            self.alpha_final = 0
            
        x_s = self._solve_core(U, s, Vh, ub, self.alpha_final)
        self.x_scaled_norm = np.linalg.norm(x_s)
        x = x_s / self.col_norms
        
        # 解爆炸保护
        if np.linalg.norm(x) > 1e10 * (self.x_scaled_norm + 1e-10):
            self.advice.append("⚠️ 警告：检测到缩放还原导致数值爆炸。尝试降低正则化强度。")
            self.alpha_final *= 1e-3
            x_s = self._solve_core(U, s, Vh, ub, self.alpha_final)
            x = x_s / self.col_norms

        # Robust 修正 (IRLS) - 仅在稠密模式下启用
        # (稀疏模式下反复重构 W*A 代价太高，通常使用 M-Estimator 的迭代加权形式，这里暂不实现)
        res_raw = self.b_raw - self.A_raw @ x
        mad = np.median(np.abs(res_raw - np.median(res_raw)))
        if mad > 1e-10:
            z = np.abs(res_raw - np.median(res_raw)) / (1.4826 * mad + 1e-15)
            if np.sum(z > 2.0) > 0:
                self.advice.append(f"[Robust] 已启动 IRLS 鲁棒修正。")
                for _ in range(10):
                    res = self.b_raw - self.A_raw @ x
                    mad_loop = np.median(np.abs(res - np.median(res)))
                    if mad_loop < 1e-12: break
                    k = 1.345 * 1.4826 * mad_loop
                    self.weights = np.ones_like(res)
                    mask = np.abs(res) > k
                    self.weights[mask] = k / np.abs(res[mask])
                    W = np.sqrt(self.weights)
                    # W * A * x = W * b
                    # 注意：这里需要重新 SVD，这在稠密下是可接受的
                    Uw, sw, Vhw, ubw = self._perform_svd(self.A * W[:, np.newaxis])
                    x_s = self._solve_core(Uw, sw, Vhw, Uw.T @ (self.b * W), self.alpha_final)
                    x = x_s / self.col_norms
        return x

    def _solve_core(self, U, s, Vh, ub, alpha):
        if alpha > 0:
            f = s**2 / (s**2 + alpha**2)
            return Vh.T @ (f / s * ub)
        else:
            mask = s > (max(self.m, self.n) * np.finfo(float).eps * s[0])
            return Vh[mask, :].T @ (ub[mask] / s[mask])

    def get_report(self, x):
        res_vec = self.b_raw - self.A @ (x * self.col_norms) if self.is_sparse else self.b_raw - self.A_raw @ x
        
        # 稀疏矩阵下，无法轻易计算全量权重的加权残差，简化处理
        if self.is_sparse:
            rel_res = np.linalg.norm(res_vec) / np.linalg.norm(self.b_raw)
            w_rel_res = rel_res # 暂无权重
            inliers = "N/A"
            conf_str = "N/A (Sparse)"
            score = 80 # 默认给高分
        else:
            rel_res = np.linalg.norm(res_vec) / np.linalg.norm(self.b_raw) if np.linalg.norm(self.b_raw) > 0 else np.linalg.norm(res_vec)
            inliers = np.sum(self.weights > 0.8)
            w_res_norm = np.linalg.norm(res_vec * np.sqrt(self.weights))
            w_rel_res = w_res_norm / np.linalg.norm(self.b_raw * np.sqrt(self.weights)) if np.linalg.norm(self.b_raw * np.sqrt(self.weights)) > 0 else w_res_norm
            
            stab = 100 - (min(50, np.log10(self.cond_raw/1e6)*10) if self.cond_raw > 1e6 else 0)
            fit = max(0, 100 - w_rel_res * 100) 
            val = (inliers / self.m) * 100
            score = int(0.4*stab + 0.3*fit + 0.3*val)
            conf = "🟢" if score > 80 else ("🟡" if score > 50 else "🔴")
            conf_str = f"{conf} {score}/100 (S:{int(stab)}|F:{int(fit)}|V:{int(val)})"

        s_min, s_med, s_max = self.diagnostics['scale_stats']
        
        report = f"""
================================================================================
                    智能数值求解器诊断报告 (Smart Solver V2.0 Sparse)
================================================================================

[阶段一: 基础体检 / DIAGNOSIS]
--------------------------------------------------------------------------------
 > 原始条件数 (Cond Raw):   {self.diagnostics.get('cond_raw', 'N/A')}
 > 缩放条件数 (Cond Scaled): {self.diagnostics.get('cond_scaled', 'N/A')}
 > 异常点探测: 有效内点 (Inliers): {inliers}

[阶段二: 证据推理 / EVIDENCE]
--------------------------------------------------------------------------------
 [i] 策略路由: {self.strategy} | Ridge Alpha: {self.alpha_final:.2e}
 [i] 诊断建议: {" ".join(self.advice)}

[最终结论 / FINAL RESULT]
--------------------------------------------------------------------------------
 > 原始相对残差 (Raw Res.): {rel_res:.4e}
 > 加权相对残差 (Wtd Res.): {w_rel_res:.4e}
 > 结果可靠性 (Reliability): {conf_str}
================================================================================
"""
        return report

def solve(A, b, verbose=True):
    solver = SmartLinearSolver(A, b)
    x = solver.solve()
    if verbose: print(solver.get_report(x))
    return x
