from helpers import *
from scalib.attacks import FactorGraph, BPState
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend import Legend
import gc
from time import time


# def gen_graph(n_shares, q=KYBER_Q):
#     graph_desc = f'''NC {q}\nVAR MULTI S\n'''
#     prop = '''PROPERTY S ='''
#     for i in range(n_shares):
#         graph_desc += f"VAR MULTI S{i}\n" #add share
#         prop +=  f" S{i}\n" if (i == n_shares-1) else f" S{i} +"
#     graph_desc += prop
#     return graph_desc


def pi_compute(n_profiling, sigma, sub_idx, model=HW, n_shares=2, q=KYBER_Q, s_set=s_range, p_s=prior_s):
    secrets = gen_secrets(n_profiling, n_coeffs=1, s_set=s_set, p_s=p_s).squeeze()
    shares = gen_shares_ext(secrets, n_shares=n_shares, sub_idx=sub_idx, q=q)
    leakages = gen_leakages(shares, sigma, model)
    pdfs = {}
    for share, share_leakage in leakages.items():
        l = np.expand_dims(share_leakage, axis=1)
        tmp = pdf_l_given_s(l, model=model, sigma=sigma, q=q)[:, model(range(q))]
        pdfs[share] = tmp/tmp.sum(axis=1, keepdims=True)
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
            bp.set_evidence(f"S{n_i}", pdfs[f"S{n_i}"])
        else:
            reverse_idx = (q - np.arange(q))%q
            pdf_si = pdfs[f"S{n_i}"] if n_i not in sub_idx else pdfs[f"S{n_i}"][:, reverse_idx]
            bp.set_evidence(f"S{n_i}", pdf_si)

    bp.bp_acyclic("S")
    resS = bp.get_distribution("S")
    return resS
def get_mi(pr_SL, s_set=s_range):
    if np.array_equal(s_set, s_range):
        pr_s = pr_SL[:, s_set]*prior_s
        pr_s = pr_s/pr_s.sum(axis=1, keepdims=True)
        ce = CE(pr_s)
        entS = ent_s()
    else:
        entS = np.log2(len(s_set))
        ce = CE(pr_SL)
    return entS-ce

def pi_curve(n_shares, sub_idx, SIGMA, N_PROFILING, REPS, q=KYBER_Q, s_set=s_range, p_s=prior_s):
    mi_holder = np.zeros_like(SIGMA)
    mode_desc = enc_repr(n_shares, sub_idx)
    f_log = f"log/pi_sasca_3329_{mode_desc}_{n_shares}shares.npy"

    # with open(f_log, "wb") as f:
    #     np.save(f, SIGMA)
    pbar = tqdm(enumerate(SIGMA), total=len(SIGMA))
    #TODO: repeat n times to reduce var
    for s_i, sigma in pbar:
        pbar.set_description_str(f"Q {q} MODE: {mode_desc} {n_shares} shares SIGMA: {sigma:0.4f} N: {N_PROFILING[s_i]}|")
        # mi = pi_compute(N_PROFILING[s_i], sigma, sub_idx, n_shares=n_shares)
        if n_shares>=2:
            resS_holder = []
            for rep in range(REPS[s_i]):
                pbar.set_postfix_str(f"REPS:{rep}/{REPS[s_i]}")
                resS = pi_compute(N_PROFILING[s_i], sigma, sub_idx, model=HW, n_shares=n_shares, q=q, s_set=s_set, p_s=p_s)
                resS_holder.append(resS)
            resS_holder = np.array(resS_holder)
            mi = get_mi(resS_holder.reshape(N_PROFILING[s_i]*REPS[s_i], q))
            pbar.set_postfix_str(f"mi: {mi:0.6f}")

        else:
            resS = pi_compute(N_PROFILING[s_i], sigma, sub_idx, model=HW, n_shares=n_shares, q=q, s_set=s_set, p_s=p_s)
            mi = get_mi(resS, s_set=s_set)
            pbar.set_postfix_str(f"mi: {mi:0.6f}")

        # with open(f_log, "ab") as f:
        #     np.save(f, [mi])
        mi_holder[s_i] = np.log10(mi)
    print_centered(f"===================PI {n_shares} shares sub_idx: {sub_idx} DONE==================")
    return mi_holder

def exp_run(q=KYBER_Q):

    N_SHARES = np.array([4])
    # line_colors = ["tab:blue", "tab:orange"]
    line_colors = ["darkmagenta", "mediumvioletred", "hotpink", "lightpink"]
    # MODES = [0, 1]
    MODES = [[0], [0, 1, 2]]
    for n_shares in N_SHARES:
        SIGMA_, _, Y_TICKS_ = gen_sigma(n_shares)
        SIGMA = SIGMA_[:5]
        Y_TICKS = Y_TICKS_[:5]
        N_PROFILING = np.ones_like(SIGMA, dtype=np.int32)*300000
        for im, mode in enumerate(MODES):
            # sub_idx = 0 if mode==0 else range(n_shares-1)
            REPS = ((Y_TICKS+n_shares+1)).astype(np.int16)
            REPS = REPS +2
            mode_desc = enc_repr(n_shares, mode)
            mi_curve = pi_curve(n_shares, mode, SIGMA, N_PROFILING, REPS)
            with open(f"log/pi_{q}_{n_shares}_{mode_desc}_small.npy", "wb") as f_:
                np.save(f_, Y_TICKS)
                np.save(f_, mi_curve)
            plt.plot(Y_TICKS, mi_curve, label=f"{enc_repr(n_shares, mode)}", color=line_colors[im])
            plt.scatter(Y_TICKS, mi_curve, color=line_colors[im])
    plt.show()

# def four_shares_modes():

def read_out():
    N_SHARES = np.array([2, 3, 4])
    line_colors = ["darkmagenta", "mediumvioletred", "hotpink", "lightpink"]
    line_st = [(0, (5, 10)),
                (0, (5, 5, 1, 5))]
    MODES = np.array(["sub", "add"])
    m_desc = ["sum", "diff"]
    fig = plt.figure()
    ax = plt.gca()
    for ni, n_shares in enumerate(N_SHARES):
        SIGMA, N_PROFILING, Y_TICKS = gen_sigma(n_shares)
        for m_i, mode in enumerate(MODES):
            sub_idx=0 if mode == "sub" else range(n_shares-1)
            f_log = f"/home/tpay/Desktop/WS/ENC/log/pi_sasca_3329_{mode}_{n_shares}shares.npy"
            with open(f_log, "rb") as f:
                sigmas = np.load(f)
                PI = []
                idx = []
                for i, si in enumerate(sigmas):
                    try:
                        pi = np.load(f)[0]
                        if pi > 0:
                            PI.append(np.log10(pi))
                            idx.append(i)
                    except:
                        pass
            ax.plot(Y_TICKS[idx], PI, color=line_colors[ni], linestyle=line_st[m_i], lw=3.5)
            with open(f"mi_sasca_{n_shares}_{enc_repr(n_shares, sub_idx)}.npy", "wb") as f:
                np.save(f, Y_TICKS[idx])
                np.save(f, PI)

    # tablelegend(ax, ncol=3, loc='best', bbox_to_anchor=(1, 1),
    #         row_labels=['sum', 'diff'],
    #         col_labels=['2 shares', '3 shares', '4 shares'],
    #         title_label=' ')
    plt.xlim(-3.1, 0.8)
    plt.ylim(-6, 0)
    plt.xlabel("$\log_{10}(\sigma^{2})$")
    plt.ylabel("$\log_{10}(MI)$")
    plt.grid()
    # plt.legend(handles=legend_lines,  fontsize=12, frameon=False)
    plt.show()
def read_out_():
    N_SHARES = np.array([2, 3, 4])
    line_colors = ["darkmagenta", "mediumvioletred", "hotpink", "lightpink"]
    line_st = [(0, (5, 10)),
                (0, (5, 5, 1, 5))]
    MODES = np.array(["sub", "add"])
    m_desc = ["sum", "diff"]
    fig = plt.figure()
    ax = plt.gca()
    for ni, n_shares in enumerate(N_SHARES):
        SIGMA, N_PROFILING, Y_TICKS = gen_sigma(n_shares)
        for m_i, mode in enumerate(MODES):
            sub_idx=0 if mode == "sub" else range(n_shares-1)
            f_log = f"/home/tpay/Desktop/WS/ENC/pi_23_{n_shares}_{enc_repr(n_shares, sub_idx)}.npy"
            print(f_log)
            with open(f_log, "rb") as f:
                pi = np.load(f)
            ax.plot(Y_TICKS, pi, color=line_colors[ni], linestyle=line_st[m_i], lw=3, label=f"{enc_repr(n_shares, sub_idx)}")
    # tablelegend(ax, ncol=3, loc='best', bbox_to_anchor=(1, 1),
    #         row_labels=['sum', 'diff'],
    #         col_labels=['2 shares', '3 shares', '4 shares'],
    #         title_label=' ')
    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(1.5)
    plt.xlim(-3.1, 0.9)
    plt.ylim(-3, 0.1)
    plt.xlabel("$\log_{10}(\sigma^{2})$")
    plt.ylabel("$\log_{10}(MI)$")
    # plt.legend(handles=legend_lines,  fontsize=12, frameon=False)
    plt.show()

def pi_line_ro():
    N_SHARES = np.array([2, 3, 4, 5, 6])
    line_st = [(0, (5, 10)),
                (0, (5, 5, 1, 5))]
    line_colors = ["forestgreen", "yellowgreen"]
    MODES = np.array(["sub", "add"])
    m_desc = ["sum", "diff"]
    fig = plt.figure()
    ax = plt.gca()
    for m_i, mode in enumerate(MODES):
        pi_line = np.zeros(len(N_SHARES))
        for ni, n_shares in enumerate(N_SHARES):
            sub_idx=0 if mode == "sub" else range(n_shares-1)
            f_log = f"/home/tpay/Desktop/WS/ENC/mi_sasca_{n_shares}_{enc_repr(n_shares, sub_idx)}.npy"
            with open(f_log, "rb") as f:
                Y_TICKS = np.load(f)
                PI = np.load(f)
            pi_line[ni] = PI[1]
        plt.plot(N_SHARES, pi_line, label=f"{m_desc[m_i]}", color=line_colors[m_i], lw=5, linestyle=line_st[m_i])
        # plt.scatter(N_SHARES, pi_line, color=line_colors[m_i], s=20)
    plt.xticks(N_SHARES)
    plt.xlabel("$d$", fontsize=12)
    plt.ylabel("$\log_{10}(MI)$")
    plt.grid()
    plt.show()
def mlp_read():
    N_TRAIN = [100, 500, 1000, 5000, 10000, 50000, 100000, 250000, 300000, 400000]
    x_ticks=N_TRAIN
    x_labels = []
    for i in N_TRAIN:
        x_labels.append(str(i) if i<1000 else f"{int(i/1000)}k")
    n_reps = 10
    n_points = 10
    d_names = ["280623_1200",  "280623_1226"]
    m_desc = ["sum", "diff"]
    fig, ax = plt.subplots()
    for i, dn in enumerate(d_names):
        pi_holder = np.zeros(len(N_TRAIN))
        for ni, n in enumerate(N_TRAIN):
            fn = f"log/mlp_pie_{dn}_on_shares_{n_points}_{n}.npy"
            pi = 0
            c = 0
            with open(fn, "rb") as f:
                for r in range(n_reps):
                    pi_curve = np.load(f)[3:]
                    # plt.plot(range(len(pi_curve)), pi_curve)
                    if pi_curve.size > 0 and pi_curve.max()> -30:
                        pi += pi_curve.max()
                        c += 1
            # plt.title(f"{n}")
            # plt.show()
                pi /= c
                pi_holder[ni] = pi

        ax.plot(x_ticks, pi_holder, label=f"{m_desc[i]}")
        with open(f"log/mlp_{dn}.npy", "wb") as f:
            np.save(f, N_TRAIN)
            np.save(f, pi_holder)
        # ax.scatter(x_ticks, pi_holder)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel("$n$")
    plt.ylabel("$PI$")
    plt.title("MLPxSASCA")
    plt.legend()
    plt.show()
    print("DONE")

# def lda_read():
    N_TRAIN = [25000, 50000, 100000, 250000, 300000, 400000]
    d_names = ["280623_1200",  "280623_1226"]
    m_desc = ["sum", "diff"]
    for di, d_name in enumerate(d_names):
        pi_holder= []
        for i_n, n_traces in enumerate(N_TRAIN):
            flog = f"log/lda_sasca_pie_{d_name}_on_shares_50_nproj10_{n_traces}.npy"
            pi = 0
            print(flog)
            with open(flog, "rb") as f:
                for rep in range(5):
                    tmp = np.load(f)
                    pi+=tmp
            pi_holder.append(pi/5)
        plt.plot(N_TRAIN, pi_holder, label=f"{m_desc[di]}")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel("$n$")
    plt.ylabel("$PI$")
    plt.title("LDAxSASCA")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    exp_run()
    # leakage_handler = Leakage_Handler("280623_1200", 195)
    # leakage_handler.get_PoI(n_pois=n_pois, mode="on_shares", model=model)
    # n_shares = leakage_handler.n_shares
    # PoI = leakage_handler.PoI.reshape(n_shares, n_pois)
    # total_n_traces = leakage_handler.n_files*leakage_handler.traces_per_file
    # traces, labels_ = leakage_handler.get_data("full_trace")
    # labels = HW(labels_[:, [0]].squeeze())
    # means_ = mean_traces(traces, labels)
    # for m in means_:
    #     plt.plot(range(195*2), m)
    # plt.show()
    # SIGMA, N_PROFILING, Y_TICKS = gen_sigma(6)
    # with open("/home/tpay/Desktop/WS/ENC/pi_6_S=-S0-S1-S2-S3-S4+S5.npy", "rb") as f:
    #     pi = np.load(f)
    # plt.plot(Y_TICKS[2:4], pi[2:4])
    # print(pi[3])
    # plt.scatter(Y_TICKS[2:4], pi[2:4])
    # with open("/home/tpay/Desktop/WS/ENC/mi_sasca_6_S=-S0-S1-S2-S3-S4+S5.npy", "wb") as f:
    #     np.save(f, Y_TICKS[2:4])
    #     np.save(f, pi[2:4])
    # with open("/home/tpay/Desktop/WS/ENC/pi_6_S=+S0+S1+S2+S3+S4+S5.npy", "rb") as f:
    #     pi = np.load(f)
    # with open("/home/tpay/Desktop/WS/ENC/mi_sasca_6_S=+S0+S1+S2+S3+S4+S5.npy", "wb") as f:
    #     np.save(f, Y_TICKS[2:4])
    #     np.save(f, pi[2:4])
    # plt.plot(Y_TICKS[2:4], pi[2:4])
    # print(pi[3])
    # plt.scatter(Y_TICKS[2:4], pi[2:4])
    # plt.show()
    # pi_line_ro()
    # read_out()
    # read_out_()
    # mlp_read()
    # SIGMA_, N_PROFILING_, Y_TICKS = gen_sigma(2)
    # SIGMA = SIGMA_[:5]
    # N_PROFILING = N_PROFILING_[:5]
    # REPS = np.ones_like(N_PROFILING, dtype=np.int8)
    # prs = np.ones(KYBER_Q)/KYBER_Q
    # mi_curve = pi_curve(n_shares=2, sub_idx=0, SIGMA=SIGMA, N_PROFILING=N_PROFILING, REPS=REPS, q=KYBER_Q, s_set=Zq, p_s=prs)
    # print(mi_curve)
    # plt.plot(Y_TICKS[:5], mi_curve, label=f"{enc_repr(2, 0)}")
    # mi_curve = pi_curve(n_shares=2, sub_idx=[0], SIGMA=SIGMA, N_PROFILING=N_PROFILING, REPS=REPS, q=KYBER_Q, s_set=Zq, p_s=prs)
    # print(mi_curve)
    # plt.plot(Y_TICKS[:5], mi_curve, label=f"{enc_repr(2, [0])}")
    # plt.legend()
    # plt.show()
    # lda_read()
    # with open("log/mlp_pie_280623_1200_on_shares_10_500.npy", "rb") as f:
    #     for i in range(10):
    #         pi = np.load(f)
    #         if len(pi) == 0:
    #             pass
    #         else:
    #             print(pi.max())
    # mlp_read()
    # read_out()
    # print(410/9)
    # exp_run()
    # p = 3329
    # Zp = np.arange(p, dtype=np.int16)
    # p_s = np.ones_like(Zp)*1/p
    # mi = pi_compute(n_profiling=400000, sigma=0.1, sub_idx=0, model=HW, n_shares=2, q=p, s_set=Zp, p_s=p_s)
    # print(mi)
    # mi = pi_compute(n_profiling=400000, sigma=0.1, sub_idx=[0], model=HW, n_shares=2, q=p, s_set=Zp, p_s=p_s)
    # print(mi)
    # mi = pi_compute(n_profiling=400000, sigma=0.1, sub_idx=[0, 1], model=HW, n_shares=3, q=p, s_set=Zp, p_s=p_s)
    # print(mi)
    # mi = pi_compute(n_profiling=400000, sigma=0.1, sub_idx=[0, 1, 2], model=HW, n_shares=4, q=p, s_set=Zp, p_s=p_s)
    # print(mi)
    # secrets = gen_secrets(20, 1)
    # shares = gen_shares_ext(secrets, n_shares=3, sub_idx=0)
    # shares = gen_shares_ext(secrets, n_shares=4, sub_idx=[0, 1])
    # shares_sanity_check(shares, secrets, 4, sub_idx=[0, 1])
    # s_set = Zq
    # p_s = np.ones_like(Zq)*(1/KYBER_Q)
    # # p_s = np.ones_like(s_range)*(1/len(s_range))
    # MODES = [0, [0], [0, 1], [0, 1, 2]]
    # n_shares = 4
    # SIGMA, N_PROFILING, Y_TICKS = gen_sigma()
    # N_PROFILING = np.ones_like(SIGMA, dtype=np.uint32)*400000
    # #
    # for sub_idx in MODES:
    #     mi = pi_curve(n_shares, sub_idx, SIGMA, N_PROFILING, q=KYBER_Q, s_set=s_set, p_s=p_s)
    #     plt.plot(SIGMA, mi, label=f"{enc_repr(n_shares, sub_idx)}")
    # plt.legend()
    # plt.show()


    # SIGMA, N_PROFILING, Y_TICKS = gen_sigma(2)
    # mi = pi_curve(2, 0, SIGMA, N_PROFILING)
    # plt.plot(Y_TICKS, mi, label=f"{enc_repr(2, 0)}")
    # mi = pi_curve(2, [0], SIGMA, N_PROFILING)
    # plt.plot(Y_TICKS, mi, label=f"{enc_repr(2, [0])}")
    # plt.legend()
    # plt.show()




    # pi_sub = pi_compute(n_profiling=100000, sigma=0.01, mode="sub", model=HW, n_shares=3)
    # pi_add = pi_compute(n_profiling=100000, sigma=0.01, mode="add", model=HW, n_shares=3)
    # pi_add_one = pi_compute(n_profiling=100000, sigma=0.01, mode="add_one", model=HW, n_shares=3)
    # print(pi_add, pi_add_one, pi_sub)
    # read_out()
    # proth_primes = np.array([13, 17, 41, 97, 113, 193, 241, 257, 353, 449, 577])
    # print("=========================Uni S==========================")
    # for P in proth_primes:
    #     Zp = np.arange(P)
    #     pi_sub = pi_compute(n_profiling=100000, sigma=0.01, mode="sub", model=HW, n_shares=2, q=P, s_set=Zp)
    #     pi_add = pi_compute(n_profiling=100000, sigma=0.01, mode="add", model=HW, n_shares=2, q=P, s_set=Zp)
    #     print(f"Q = {P} pi_sub = {pi_sub} pi_add = {pi_add}")
    # print("=========================CBD S==========================")
    # for P in proth_primes:
    #     Zp = np.arange(P)
    #     pi_sub = pi_compute(n_profiling=100000, sigma=0.01, mode="sub", model=HW, n_shares=2, q=P, s_set=s_range)
    #     pi_add = pi_compute(n_profiling=100000, sigma=0.01, mode="add", model=HW, n_shares=2, q=P, s_set=s_range)
    #     print(f"Q = {P} pi_sub = {pi_sub} pi_add = {pi_add}")
    # SIGMA, N_PROFILING, Y_TICKS = gen_sigma(4)
    # PI = np.zeros_like(SIGMA)
    # with open("/home/tpay/Desktop/WS/ENC/log/pi_sasca_3329_sub_4shares-.npy", "rb") as f:
    #     sigma = np.load(f)
    #     for i, si in enumerate(SIGMA):
    #         try:
    #             pi = np.load(f)[0]
    #             if pi > 0:
    #                 PI[i] += pi
    #         except:
    #             pass
    # with open("/home/tpay/Desktop/WS/ENC/log/pi_sasca_3329_sub_4shares_.npy", "rb") as f:
    #     sigma = np.load(f)
    #     for i, si in enumerate(SIGMA):
    #         try:
    #             pi = np.load(f)[0]
    #             if pi > 0:
    #                 PI[i] += pi
    #         except:
    #             pass
    # with open("/home/tpay/Desktop/WS/ENC/log/pi_sasca_3329_sub_4shares__.npy", "rb") as f:
    #     sigma = np.load(f)
    #     for i, si in enumerate(SIGMA):
    #         try:
    #             pi = np.load(f)[0]
    #             if pi > 0:
    #                 PI[i] += pi
    #         except:
    #             pass
    # with open("/home/tpay/Desktop/WS/ENC/log/pi_sasca_3329_sub_4shares___.npy", "rb") as f:
    #     sigma = np.load(f)
    #     for i, si in enumerate(SIGMA):
    #         try:
    #             pi = np.load(f)[0]
    #             if pi > 0:
    #                 PI[i] += pi
    #         except:
    #             pass
    # with open("/home/tpay/Desktop/WS/ENC/log/pi_sasca_3329_sub_4shares.npy", "wb") as f:
    #     np.save(f, Y_TICKS[:10])
    #     PI = PI/4
    #     for pi in PI[:10]:
    #
    #         np.save(f, [pi])
    # plt.plot(Y_TICKS, PI/4)
    # plt.show()
    # exp_run()
    # st = time()
    # pi_compute(n_profiling=50000, sigma=1, mode="sub", model=HW, n_shares=2, q=KYBER_Q, s_set=s_range)
    # print(time()-st)

    # print(s2)
    # Q = 6011
    # ZQ = np.arange(Q)
    # pi = pi_est(n_profiling=75000, sigma=0.1, mode="sub", model=HW, q=Q, s_set=ZQ)
    # print(pi)
    # pi = pi_est(n_profiling=75000, sigma=0.1, mode="add", model=HW, q=Q, s_set=ZQ)
    # print(pi)
