from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scalib.metrics import SNR
from scalib.attacks import SASCAGraph, FactorGraph, BPState

from scalib.modeling import MultiLDA, LDAClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import multivariate_normal
import gc
from time import time
from tqdm import trange
from math import ceil
from helpers import *
def NLLLoss(preds, targets):
    # PRIOR = np.array([0.375, 0.25, 0.0625, 0.0625, 0.25])
    preds[np.where(~np.isfinite(preds))] = 1e-100
    preds[np.where(preds < 1e-100)] = 1e-100
    preds_log = np.log2(preds)
    n_classes, p_L = np.unique(targets, return_counts=True)
    ce = 0
    for i in n_classes:
        tmp = preds[targets==i, i]
        ce += tmp.mean()*p_L[i]

    # ce = 0
    # # sec, sec_count = np.unique(targets, return_counts=True)
    # for s_i in range(5):
    #     tmp = preds_log[targets==s_i, s_i]
    #     ce += tmp.mean()*PRIOR[s_i]
    return -ce/len(targets)
def share_model(L_train, L_val, label_train, n_proj, model=ID):
    ns = L_train.shape[1]
    lda = LDAClassifier(KYBER_Q, n_proj, ns)
    if len(L_train) > 500000:
        n_chunks = ceil(len(L_train)/500000)
        L_train_chunks = np.array_split(L_train, n_chunks, axis=0)
        s_train_chunks = np.array_split(label_train, n_chunks, axis=0)
        for i_c in trange(n_chunks, desc="LDA CHUNKs"):
            lda.fit_u(L_train_chunks[i_c], s_train_chunks[i_c])
    else:
        lda.fit_u(L_train, label_train)
    lda.solve()
    predicted_proba = lda.predict_proba(L_val)
    return predicted_proba
def share_model_gt(L_train, L_val, label_train, model=ID):

    ns = L_train.shape[1]
    templates = GT(Nk=KYBER_Q, Nd=ns)
    templates.fit(L_train, label_train)
    means, variances = templates.get_template()
    for q in trange(KYBER_Q, desc=f"q PDF"):
        print(np.all(np.linalg.eigvals(variances[q]) > 0))
        print(variances[q].shape)
        # print(multivariate_normal.pdf(L_val, means[q], variances[q]).shape)
def sasca(pdf_shares, m_flag=0):
    S = np.array([0, 1, 2, 3327, 3328])
    P_S = np.array([0.375, 0.25, 0.0625, 0.0625, 0.25])
    n_shares = len(pdf_shares)
    total_N = pdf_shares[0].shape[0]
    pSL = np.zeros((total_N, len(S)))
    for s_i, s in tqdm(enumerate(S), total=len(S), desc="SASCA s|"):
        idx_share = np.zeros((n_shares, KYBER_Q**(n_shares-1)), dtype=np.uint16)
        for n_i in range(n_shares-1):
            idx_share[n_i] = np.arange(KYBER_Q)


        # acc_shares = idx_share[:(n_shares-1)].sum(axis=1)
        # print("acc share ")
        # print(acc_shares)

        idx_share[n_shares-1] = (s - idx_share[0] +KYBER_Q)%KYBER_Q

        p_si_l = np.ones_like(pdf_shares[0])
        for n_i in range(n_shares):
            p_si_l = p_si_l*pdf_shares[n_i][:, idx_share[n_i]]
        pSL[range(total_N), s_i] = p_si_l.sum(axis=1)
    pSL *= P_S

    return pSL/(pSL.sum(axis=1, keepdims=True))
def share_model_check(d_name):
    lh = Leakage_Handler(d_name, 195)
    lh.n_files = 200
    lh.get_PoI(n_pois=50, mode="on_shares", model=ID, keep=True)
    PoI = lh.PoI.reshape(lh.n_shares, 50)
    n_shares = lh.n_shares
    traces_, labels_ = lh.get_data("full_trace")
    traces_ = traces_.astype(np.int16)
    total_N = len(traces_)
    N_PROFILING = [50000, 100000, 200000, 300000, 500000, 700000, 800000, 1000000, 1500000]
    # , 800000, 1000000, 1500000]
    n_val = 200000
    pi_share = np.zeros((2, len(N_PROFILING)))
    PI = np.zeros((len(N_PROFILING)))
    for npi, n_p in enumerate(N_PROFILING):
        pdfs = []
        trace_train = traces_[:n_p]
        trace_val = traces_[-n_val:]
        labels_val = labels_[-n_val:]
        secret_val = labels_val[:, [2]].squeeze()
        secret_val = (fix_s(secret_val)%5).astype(np.int8)
        for n_i in trange(lh.n_shares, desc="PDFSHARE|"):
            # traces = traces_[:, PoI[n_i] ].astype(np.int16)
            labels = labels_[:, [n_i]].astype(np.uint16).squeeze()
            label_train = labels[:n_p]
            label_val = labels[-n_val:]
            p_lda = share_model(trace_train, trace_val, label_train, 50, model=ID)

            pdfs.append(p_lda.copy())

            # loss = CE_(p_lda, label_val)
    #         pi = np.log2(KYBER_Q) - loss
    #         print(f"============N_P {n_p} SHARE : {n_i} CE {loss} PI: {pi}==============")
    #         pi_share[n_i, npi] = pi
    # with open(f"LDA_PI_SHARE.npy", "wb") as f:
    #     np.save(f, pi_share)
    # plt.plot(pi_share.T)
    # plt.show()
            # print_centered(f" N_PROFILING: {n_p} SHARE {n_i+1} CE {loss} PI: {pi}")
        # graph_desc = gen_graph(n_shares=lh.n_shares)
        # graph = FactorGraph(graph_desc)
        # bp = BPState(graph, 300000)
        # for i in range(lh.n_shares):
        #     if lh.m_flag==10:
        #         bp.set_evidence(f"S{i}", pdfs[i].astype(np.float64))
        #     else:
        #         reverse_idx = (KYBER_Q - np.arange(KYBER_Q))%KYBER_Q
        #         pdf_si = pdfs[i] if i== lh.n_shares-1 else pdfs[i][:, reverse_idx]
        #         bp.set_evidence(f"S{i}", pdf_si.astype(np.float64))
        #
        # bp.bp_acyclic("S")
        # resS = bp.get_distribution("S")
        pr_s = sasca(pdfs, m_flag=0)
        # print(pr_s.shape)
        # print(pr_s.sum(axis=1))
        # exit()

        # pr_s = np.zeros((n_val, 5))
        #
        # pr_s[:, range(5)] = resS[:, [0, 1, 2, -2, -1]]
        # pri_s = np.array([0.375, 0.25, 0.0625, 0.0625, 0.25])
        # pr_s *= pri_s
        # pr_s = pr_s/pr_s.sum(axis=1, keepdims=True)
        # pr_s[np.where(~np.isfinite(pr_s))] = 1e-100
        # pr_s[np.where(pr_s < 1e-100)] = 1e-100
        entS = ent_s(prior_s)
        # # ce = NLLLoss(pr_s, secret_val)
        ce = CE(pr_s)
        # ce = CE_(pr_s, secret_val)

        print_centered(f"N_PROFILING: {n_p} N_VAL: {n_val} CE {ce} H: {entS} PI: {entS - ce}")
        PI[npi] = entS - ce
    plt.plot(N_PROFILING, PI)
    plt.xlabel("N_PROFILING")
    plt.ylabel("PI")
    plt.show()

def fix_s(s):
    s_ = s.copy().astype(np.int16)
    s_[s_==3328] = -1
    s_[s_==3327] = -2
    return s_

def lda_sasca(leakage_handler, traces, labels, n_profiling, n_proj=10, model=None,pbar=None):
    total_n_traces = leakage_handler.n_files*leakage_handler.traces_per_file
    n_pois = leakage_handler.n_pois
    PoI = leakage_handler.PoI.reshape(leakage_handler.n_shares, n_pois)
    pdfs = []

    train_idx, val_idx = under_sample(total_n_traces, n_profiling, labels, leakage_handler.n_shares, n_val=100000)
    s_val = labels[val_idx, 2]
    s_val = fix_s(s_val)
    s_val = (s_val%5).astype(np.int8)
    for i in trange(leakage_handler.n_shares, desc="PDFSHARE", leave=False):
        traces_i = traces[:, PoI[i]]
        Li_train = traces_i[train_idx[i]]
        si_train = labels[train_idx[i], i]
        L_val = traces_i[val_idx]
        si_val = labels[val_idx, i]
        pdf_share = share_model(Li_train, L_val, si_train, n_proj=n_proj)
        # pdf_share = (pdf_share.T / np.sum(pdf_share,axis=1)).T
        pdfs.append(pdf_share.copy())

    # integrate to graph
    graph_desc = gen_graph(n_shares=leakage_handler.n_shares, q=KYBER_Q)
    graph = FactorGraph(graph_desc)
    bp = BPState(graph, 100000)
    for i in range(leakage_handler.n_shares):
        if leakage_handler.m_flag==10:
            bp.set_evidence(f"S{i}", pdfs[i].astype(np.float64))
        else:
            reverse_idx = (KYBER_Q - np.arange(KYBER_Q))%KYBER_Q
            pdf_si = pdfs[i] if i== leakage_handler.n_shares-1 else pdfs[i][:, reverse_idx]
            bp.set_evidence(f"S{i}", pdf_si.astype(np.float64))

    bp.bp_acyclic("S")
    resS = bp.get_distribution("S")
    pr_s = resS[:, s_range]*prior_s
    pr_s = pr_s/pr_s.sum(axis=1, keepdims=True)
    ce = CE(pr_s)
    entS = ent_s()
    if pbar:
        pbar.set_postfix_str(f"PI: {entS - ce}")

    return entS - ce


def pi_curve_wn(d_name, N_TRAIN, n_pois=50, n_proj=5, model=ID, mode="on_shares"):
    leakage_handler = Leakage_Handler(d_name, 195)
    leakage_handler.n_files = 200
    leakage_handler.get_PoI(n_pois=n_pois, mode="on_shares", model=model, keep=True)
    n_shares = leakage_handler.n_shares
    traces, labels_ = leakage_handler.get_data("full_trace")
    traces = traces.astype(np.int16)
    labels = labels_.astype(np.uint16)
    pi_holder = np.zeros(len(N_TRAIN))
    ns_pbar = tqdm(enumerate(N_TRAIN), total=len(N_TRAIN))
    for i_n, n_traces in ns_pbar:
        ns_pbar.set_description_str(f"{d_name} N_TRACES: {n_traces}")
        pi = lda_sasca(leakage_handler, traces, labels, n_traces, n_proj=n_proj, model=ID, pbar=ns_pbar)
        ns_pbar.set_postfix_str(f"PI {pi:0.4f}")
        pi_holder[i_n] = pi
    return pi_holder



if __name__ == '__main__':
    # N_PROFILING = [50000, 100000, 200000, 300000, 400000, 500000, 600000]
    # pi_curve = pi_curve_wn("280623_1200", N_PROFILING)
    # plt.plot(N_PROFILING, pi_curve)
    # plt.show()
    share_model_check("280623_1200")
    # N_TRAIN = [50000, 100000, 200000, 300000, 500000, 700000, 800000, 1000000]
    # with open("LDA_PI_SHARE.npy", "rb") as f:
    #     pi_shares = np.load(f)
    #     for i, pi in enumerate(pi_shares):
    #         plt.plot(N_TRAIN, pi, label=f"share {i+1}")
    # plt.legend()
    # plt.xlabel("N_PROFILING")
    # plt.ylabel("PI")
    # plt.show()
    # leakage_handler = Leakage_Handler("280623_1200", 195)
    # leakage_handler.n_files = 200
    # leakage_handler.get_PoI(n_pois=5, mode="on_sec", model=ID, keep=True, display=True)
    # with open("log/280623_1200_snr_on_sec.npy", "wb") as f:
    #     np.save(f, leakage_handler.snr)
    # leakage_handler = Leakage_Handler("280623_1226", 195)
    # leakage_handler.n_files = 200
    # leakage_handler.get_PoI(n_pois=5, mode="on_sec", model=ID, keep=True, display=True)
    # with open("log/280623_1226_snr_on_sec.npy", "wb") as f:
        # np.save(f, leakage_handler.snr)
    # n_shares = leakage_handler.n_shares
    # traces, labels_ = leakage_handler.get_data("full_trace")
    # traces = traces.astype(np.int16)
    # labels = labels_[:, [0, 1]].astype(np.uint16)
    # total_n_traces = leakage_handler.n_files*leakage_handler.traces_per_file
    # n_pois = leakage_handler.n_pois
    # PoI = leakage_handler.PoI.reshape(leakage_handler.n_shares, n_pois)
    # traces_i = traces[:, PoI[0]]
    # labels_i = labels[:, [0]].squeeze()
    # n_proj = 10

    # train_idx, val_idx = under_sample(total_n_traces, n_profiling, labels, leakage_handler.n_shares)
    # Li_train = traces_i[train_idx[0]]
    # si_train = labels[train_idx[0], 0]
    # L_val = traces_i[val_idx]
    # si_val = labels[val_idx, 0]
    # Li_train, L_val, s_train, s_val = train_test_split(traces_i, labels_i, test_size=0.5)


    # share_model_gt(Li_train, L_val, s_train, n_proj, model=ID)
    # print(pdf1[:5])
    # print_centered(f"\n=========PI share 1 {len(Li_train)} {pdf1.shape} {np.log2(KYBER_Q)-CE(pdf1)} {CE(pdf1)}==============\n")
    # n_profiling = int(total_n_traces*0.8)
    # Li_train, L_val, s_train, s_val = train_test_split(traces_i, labels_i, test_size=0.6)
    # # train_idx, val_idx = under_sample(total_n_traces, n_profiling, labels, leakage_handler.n_shares)
    # # Li_train = traces_i[train_idx[0]]
    # # si_train = labels[train_idx[0], 0]
    # # L_val = traces_i[val_idx]
    # # si_val = labels[val_idx, 0]
    #
    #
    # pdf = share_model(Li_train, L_val, s_train, n_proj, model=ID)
    # # print(pdf[:5])
    # print_centered(f"\n=========PI share 1 {len(Li_train)} {pdf.shape} {np.log2(KYBER_Q)-CE(pdf)} {CE(pdf)}==============\n")
    # print(pdf-pdf1)
    # N_TRAIN = [50000, 100000, 250000, 400000, 500000, 600000,  700000, 800000, 900000,1000000,1250000]
    # pi = pi_curve_wn("280623_1200", N_TRAIN, n_pois=10, n_proj=10, model=ID, mode="on_shares")
    # with open(f"lda_sasca_280623_1200.npy", "wb") as f:
    #     np.save(f, N_TRAIN)
    #     np.save(f, pi)
    # plt.plot(N_TRAIN, pi, label=f"S=X1+X2")
    # pi = pi_curve_wn("280623_1226", N_TRAIN, n_pois=10, n_proj=10, model=ID, mode="on_shares")
    # with open(f"lda_sasca_280623_1200.npy", "wb") as f:
    #     np.save(f, N_TRAIN)
    #     np.save(f, pi)
    # plt.plot(N_TRAIN, pi, label=f"S=X2-X1")
    # plt.legend()
    # plt.show()
    # x = np.array([1, 2, 3328, 3327, 0])
    # y = (x - 3329)
    # print(y)
    # print_centered(f"==========MLP_SASCA PROCESS ID MODEL 2 SHARES===============")
    # pi_sub = mlp_shares("280623_1200", model=ID)
    # pi_add = mlp_shares("280623_1226", model=ID)
    # print_centered("===============MLPSASCA 2 shares share_val model==============")
    # print_centered(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    # print_centered("==================================")
    #pi_sub = mlp_sasca("200623_1211")
    #pi_add = mlp_sasca("200623_1225")
    #print_centered("===============MLPSASCA 3 shares share_val model==============")
    #print_centered(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    #print_centered("==================================")
    # print_centered(f"==========MLP_SASCA PROCESS ID MODEL 2 SHARES===============")
    # pi_sub = mlp_shares("280623_1200", model=HW)
    # pi_add = mlp_shares("280623_1226", model=HW)
    # print_centered("===============MLPSASCA 2 shares HW model==============")
    # print_centered(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    # print_centered("==================================")
    #pi_sub = mlp_sasca("200623_1211", model="HW")
    #pi_add = mlp_sasca("200623_1225", model="HW")
    #print_centered("===============MLPSASCA 3 shares HW model==============")
    #print_centered(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    #print_centered("==================================")

    # y_leakage = np.random.rand(2, 256) # this might come from an LDA
    # print(y_leakage.shape)
    # y_leakage = y_leakage / y_leakage.sum(axis=1, keepdims=True)
    # print(y_leakage.shape, y_leakage.sum(axis=1))

    # share_model("190623_1600", 10, 0)
