from helpers import *
from scalib.attacks import FactorGraph, BPState
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import gc
from time import time

def int_l(lmin=-2, lmax=7, dense=10):
    L_len = int(lmax-lmin)
    int_l0 =  np.linspace(lmin, lmax, num=L_len*dense, endpoint=True)
    int_l1 =  np.linspace(lmin, lmax, num=L_len*dense, endpoint=True)
    int_L = np.meshgrid(int_l0, int_l1)
    L0 = int_L[0].flatten()
    L1 = int_L[1].flatten()
    return L0, L1

def gen_graph(n_shares, q=KYBER_Q):
    graph_desc = f'''NC {q}\nVAR MULTI S\n'''
    prop = '''PROPERTY S ='''
    for i in range(n_shares):
        graph_desc += f"VAR MULTI S{i}\n" #add share
        prop +=  f" S{i}\n" if (i == n_shares-1) else f" S{i} +"
    graph_desc += prop
    return graph_desc



def pi_compute(n_profiling, sigma, sub_idx, model=HW, n_shares=2, q=KYBER_Q, s_set=s_range, p_s=prior_s):
    secrets = gen_secrets(n_profiling, n_coeffs=1, s_set=s_set, p_s=p_s).squeeze()
    shares = gen_shares_ext(secrets, n_shares=n_shares, sub_idx=sub_idx, q=q)
    lmin = - sigma*5
    lmax = 11 + sigma*5
    leakages = int_l(lmin, lmax, dense=10)
    pdfs = []
    for n_s in range(n_shares):
        l = np.expand_dims(leakages[n_s], axis=1)
        tmp = pdf_l_given_s(l, model=model, sigma=sigma, q=q)[:, model(range(q))]
        pdfs.append(tmp/tmp.sum(axis=1, keepdims=True))
    del secrets
    del shares
    del leakages
    del tmp
    gc.collect()


    graph_desc = gen_graph(n_shares, q=q)
    graph = FactorGraph(graph_desc)
    bp = BPState(graph, n_profiling)
    for n_i in range(n_shares):
        if isinstance(sub_idx, int):
            bp.set_evidence(f"S{n_i}", pdfs[n_i])
        else:
            reverse_idx = (q - np.arange(q))%q
            pdf_si = pdfs[n_i] if n_i not in sub_idx else pdfs[n_i][:, reverse_idx]
            bp.set_evidence(f"S{n_i}", pdf_si)

    bp.bp_acyclic("S")
    resS = bp.get_distribution("S")
    if np.array_equal(s_set, s_range):
        pr_s = np.zeros((n_profiling, len(s_set)))
        pr_s[:, s_set] = resS[:, s_set]*prior_s
        pr_s = pr_s/pr_s.sum(axis=1, keepdims=True)
        ce = CE(pr_s)
        entS = ent_s()
    else:
        entS = np.log2(len(s_set))
        ce = CE(resS)
    return entS-ce

def pi_curve(n_shares, sub_idx, SIGMA, N_PROFILING, q=KYBER_Q, s_set=s_range, p_s=prior_s):
    mi_holder = np.zeros_like(SIGMA)
    f_log = f"log/pi_sasca_3329_{sub_idx}_{n_shares}shares.npy"

    with open(f_log, "wb") as f:
        np.save(f, SIGMA)
    pbar = tqdm(enumerate(SIGMA), total=len(SIGMA))
    #TODO: repeat n times to reduce var
    for s_i, sigma in pbar:
        pbar.set_description_str(f"MODE: {sub_idx} {n_shares} shares SIGMA: {sigma:0.4f} N: {N_PROFILING[s_i]}|")
        # mi = pi_compute(N_PROFILING[s_i], sigma, sub_idx, n_shares=n_shares)
        mi = pi_compute(N_PROFILING[s_i], sigma, sub_idx, model=HW, n_shares=n_shares, q=KYBER_Q, s_set=s_set, p_s=p_s)
        pbar.set_postfix_str(f"mi: {mi:0.6f}")
        with open(f_log, "ab") as f:
            np.save(f, [mi])
        mi_holder[s_i] = np.log10(mi)
    print_centered(f"===================PI {n_shares} shares sub_idx: {sub_idx} DONE==================")
    return mi_holder

if __name__ == '__main__':
    SIGMA_, _, Y_TICKS_ = gen_sigma(2)
    SIGMA = SIGMA_[:-3]
    Y_TICKS = Y_TICKS_[:-3]
    N_PROFILING = (11 + 10*SIGMA).astype(np.uint32)
    N_PROFILING = (N_PROFILING*10)**2

    mi = pi_curve(2, 0, SIGMA, N_PROFILING)
    plt.plot(Y_TICKS, mi, label=f"{enc_repr(2, 0)}")
    mi = pi_curve(2, [0], SIGMA, N_PROFILING)
    plt.plot(Y_TICKS, mi, label=f"{enc_repr(2, [0])}")
    plt.legend()
    plt.show()
