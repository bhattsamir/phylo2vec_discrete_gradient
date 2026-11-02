import numpy as np
rng = np.random.default_rng(1)
def gtr(S6, pi):
    S = np.zeros((4,4)); iu = np.triu_indices(4,1); S[iu] = S6; S = S + S.T
    Q = S * pi[None,:]; np.fill_diagonal(Q, 0); Q -= np.diag(Q.sum(1))
    mu = -(pi*np.diag(Q)).sum(); return Q/mu
def Pgamma(Q, pi, t, a):
    Dh = np.diag(np.sqrt(pi)); R = Dh @ Q @ np.linalg.inv(Dh); R = (R+R.T)/2
    lam, V = np.linalg.eigh(R); m = (1 - t*lam/a)**(-a)
    return np.linalg.inv(Dh) @ V @ np.diag(m) @ V.T @ Dh, lam
sv = lambda M: np.sort(np.linalg.svd(M, compute_uv=False))[::-1]
mx = {k: 0.0 for k in "1a_eig 1a_sv_bias mode0_code_bias mode0_paper_bias 1b_sym 1b_transpose 1b_contract 2_deflate 3_est 3_with_half 5_rows 5_neg".split()}
worst = {}
for trial in range(2000):
    pi = rng.dirichlet([2,2,2,2]); S6 = rng.lognormal(0,0.7,6); a = rng.uniform(0.2,3); t = rng.uniform(0.01,1.5)
    Q = gtr(S6, pi); P, lam = Pgamma(Q, pi, t, a)
    K = np.diag(pi) @ P                      # population joint matrix, Eq. (8)
    D = np.diag(pi); eigP = np.sort(np.linalg.eigvals(P).real)[::-1]
    Lorig = D**0.5 @ K @ np.linalg.inv(D)**1.5
    mx["1a_eig"] = max(mx["1a_eig"], np.abs(np.sort(np.linalg.eigvals(Lorig).real)[::-1]-eigP).max())
    mx["1a_sv_bias"] = max(mx["1a_sv_bias"], np.abs(sv(Lorig)-eigP).max())
    Fa, Fb = K.sum(1), K.sum(0)
    A = lambda F, p, q: np.diag(F**p) @ K @ np.diag(F**q)
    code0 = 0.5*(A(Fa,.5,-1.5) + A(Fb,-1.5,.5)); paper0 = 0.5*(A(Fa,.5,-1.5) + A(Fb,.5,-1.5))
    trL = -np.trace(Q)
    est = lambda s, start: sum(a*(min(x,1)**(-1/a)-1) for x in s[start:])/trL
    mx["mode0_code_bias"] = max(mx["mode0_code_bias"], abs(est(sv(code0),0)-t)/t)
    mx["mode0_paper_bias"] = max(mx["mode0_paper_bias"], abs(est(sv(paper0),0)/2*1 - t)/t)  # paper Eq6 has 1/2
    Ls = np.diag(Fa**-.5) @ K @ np.diag(Fb**-.5)
    mx["1b_sym"] = max(mx["1b_sym"], np.abs(Ls-Ls.T).max())
    Lt = Ls - np.outer(np.sqrt(Fa), np.sqrt(Fb)); s = sv(Lt)
    d_hat = est(np.r_[1, s[:3]], 1)
    mx["3_est"] = max(mx["3_est"], abs(d_hat-t)/t); mx["3_with_half"] = max(mx["3_with_half"], abs(d_hat/2-t)/t)
    mx["5_rows"] = max(mx["5_rows"], np.abs(P.sum(1)-1).max()); mx["5_neg"] = min(mx["5_neg"], P.min())
# Claims 1b/2 on arbitrary NON-stationary random K (not from a model)
for trial in range(5000):
    K = rng.dirichlet(np.ones(16)*rng.uniform(0.05,3)).reshape(4,4); Fa, Fb = K.sum(1), K.sum(0)
    L = np.diag(Fa**-.5) @ K @ np.diag(Fb**-.5); Lba = np.diag(Fb**-.5) @ K.T @ np.diag(Fa**-.5)
    mx["1b_transpose"] = max(mx["1b_transpose"], np.abs(Lba - L.T).max())
    mx["1b_contract"] = max(mx["1b_contract"], sv(L)[0]-1)
    Lt = L - np.outer(np.sqrt(Fa), np.sqrt(Fb))
    mx["2_deflate"] = max(mx["2_deflate"], np.abs(np.sort(np.r_[sv(L)[1:],0])[::-1] - sv(Lt)).max())
for k,v in mx.items(): print(f"{k:18s} {v:.3e}")
