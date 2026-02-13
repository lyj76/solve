import numpy as np
from scipy import optimize

class SmartLinearSolver:
    def __init__(self, A, b):
        self.A_raw = np.asarray(A, dtype=float)
        self.b_raw = np.asarray(b, dtype=float).flatten()
        self.m, self.n = self.A_raw.shape
        
        # 1. 原始条件数
        s_raw = np.linalg.svd(self.A_raw, compute_uv=False)
        self.cond_raw = s_raw[0] / s_raw[-1] if s_raw[-1] > 0 else float('inf')
        
        # 2. 自动列缩放
        self.col_norms = np.linalg.norm(self.A_raw, axis=0)
        med_norm = np.median(self.col_norms) if len(self.col_norms) > 0 else 1.0
        self.s_floor = max(1e-12, med_norm * 1e-10)
        self.col_norms = np.maximum(self.col_norms, self.s_floor)
        
        self.A = self.A_raw / self.col_norms
        self.b = self.b_raw
        
        self.U, self.s, self.Vh = None, None, None
        self.u_dot_b = None
        self.alpha_final = 0
        self.diagnostics = {}
        self.weights = np.ones(self.m)
        self.advice = []
        self.x_scaled_norm = 0
        
    def _perform_svd(self, A_current=None):
        if A_current is None:
            if self.U is None:
                self.U, self.s, self.Vh = np.linalg.svd(self.A, full_matrices=False)
                self.u_dot_b = self.U.T @ self.b
            return self.U, self.s, self.Vh, self.u_dot_b
        else:
            U, s, Vh = np.linalg.svd(A_current, full_matrices=False)
            return U, s, Vh, U.T @ self.b

    def diagnose(self):
        U, s, Vh, ub = self._perform_svd()
        cond_scaled = s[0] / s[-1] if s[-1] > 0 else float('inf')
        tol = max(self.m, self.n) * np.finfo(float).eps * s[0]
        rank = np.sum(s > tol)
        
        self.diagnostics = {
            "shape": (self.m, self.n),
            "cond_raw": self.cond_raw,
            "cond_scaled": cond_scaled,
            "rank": rank,
            "scale_stats": (np.min(self.col_norms), np.median(self.col_norms), np.max(self.col_norms))
        }
        
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
        U, s, Vh, ub = self._perform_svd()
        if "Ridge" in self.strategy:
            self.alpha_final = self.find_alpha_hybrid(U, s, Vh, ub)
        else:
            self.alpha_final = 0
            
        x_s = self._solve_core(U, s, Vh, ub, self.alpha_final)
        self.x_scaled_norm = np.linalg.norm(x_s)
        x = x_s / self.col_norms
        
        # Robust 修正
        res_raw = self.b_raw - self.A_raw @ x
        mad = np.median(np.abs(res_raw - np.median(res_raw)))
        if mad > 1e-10:
            # 提高灵敏度: 2.0 代替 3.5
            z = np.abs(res_raw - np.median(res_raw)) / (1.4826 * mad + 1e-15)
            if np.sum(z > 2.0) > 0:
                self.advice.append(f"[Robust] 已启动 IRLS 鲁棒修正。")
                for _ in range(10):
                    res = self.b_raw - self.A_raw @ x
                    mad_loop = np.median(np.abs(res - np.median(res)))
                    if mad_loop < 1e-12: break
                    k = 1.345 * 1.4826 * mad_loop
                    self.weights = np.ones_like(res)
                    # 鲁棒权重应用
                    mask = np.abs(res) > k
                    self.weights[mask] = k / np.abs(res[mask])
                    
                    W = np.sqrt(self.weights)
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
        res_vec = self.b_raw - self.A_raw @ x
        rel_res = np.linalg.norm(res_vec) / np.linalg.norm(self.b_raw) if np.linalg.norm(self.b_raw) > 0 else np.linalg.norm(res_vec)
        inliers = np.sum(self.weights > 0.8)
        w_res_norm = np.linalg.norm(res_vec * np.sqrt(self.weights))
        w_rel_res = w_res_norm / np.linalg.norm(self.b_raw * np.sqrt(self.weights)) if np.linalg.norm(self.b_raw * np.sqrt(self.weights)) > 0 else w_res_norm
        stab = 100 - (min(50, np.log10(self.cond_raw/1e6)*10) if self.cond_raw > 1e6 else 0)
        fit = max(0, 100 - w_rel_res * 100) 
        val = (inliers / self.m) * 100
        score = int(0.4*stab + 0.3*fit + 0.3*val)
        conf = "🟢" if score > 80 else ("🟡" if score > 50 else "🔴")
        s_min, s_med, s_max = self.diagnostics['scale_stats']
        
        report = f"""
================================================================================
                    智能数值求解器诊断报告 (Smart Solver V1.4.1)
================================================================================

[阶段一: 基础体检 / DIAGNOSIS]
--------------------------------------------------------------------------------
 > 原始条件数 (Cond Raw):   {self.cond_raw:.2e}
 > 缩放条件数 (Cond Scaled): {self.diagnostics['cond_scaled']:.2e}
 > 异常点探测: 有效内点 (Inliers): {inliers}/{self.m} (占比: {inliers/self.m*100:.1f}%)

[阶段二: 证据推理 / EVIDENCE]
--------------------------------------------------------------------------------
 [i] 策略路由: {self.strategy} | Ridge Alpha: {self.alpha_final:.2e}
 [i] 诊断建议: {" ".join(self.advice)}

[最终结论 / FINAL RESULT]
--------------------------------------------------------------------------------
 > 原始相对残差 (Raw Res.): {rel_res:.4e}
 > 加权相对残差 (Wtd Res.): {w_rel_res:.4e}
 > 结果可靠性 (Reliability): {conf} {score}/100 (S:{int(stab)}|F:{int(fit)}|V:{int(val)})
================================================================================
"""
        return report

def solve(A, b, verbose=True):
    solver = SmartLinearSolver(A, b)
    x = solver.solve()
    if verbose: print(solver.get_report(x))
    return x
