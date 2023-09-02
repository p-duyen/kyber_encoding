from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Callback
from lightning.pytorch import loggers as pl_loggers
from sklearn.model_selection import train_test_split
from scalib.metrics import SNR
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scalib.attacks import SASCAGraph, FactorGraph, BPState
from lightning.pytorch.callbacks import ModelCheckpoint
import gc
from time import time
from tqdm import trange
from mlp_arch import *
from helpers import *
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
torch.autograd.set_detect_anomaly(True)
class _EarlyStopping(EarlyStopping, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
class _ModelCheckpoint(ModelCheckpoint, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



def share_model(d_name, L_train, L_val, s_train, s_val, share_i, load_saved=0, model=ID):

    # L_train, L_val, s_train, s_val = train_test_split(traces, labels, test_size=0.25, random_state=42, shuffle=None)
    train_batch = L_train.shape[0]//5 if len(L_train)<10000 else L_train.shape[0]//50
    val_batch = s_val.shape[0]//20

    valset = LeakageData(L_val, s_val)
    valdata = DataLoader(valset, batch_size=val_batch,num_workers=20, shuffle=False)


    chkpt_dir_path = f"ENC_{d_name}_{len(L_train)}_checkpoint"
    if model is ID:
        mlp = MLP(in_dim=L_train.shape[-1], out_dim=KYBER_Q)
    else:
        mlp = MLP(in_dim=L_train.shape[-1], out_dim=HW_Q)
    filename=f"share_{share_i}"
    if load_saved:
        # checkpoint = torch.load(f"{chkpt_dir_path}/{filename}.ckpt")
        # print(checkpoint['module_arguments'])
        saved_model = mlp.load_from_checkpoint(f"{chkpt_dir_path}/{filename}.ckpt", in_dim=L_train.shape[-1], out_dim=KYBER_Q)
        saved_model.eval()
        pred_data = DataLoader(valset, batch_size=L_val.shape[0],num_workers=10, shuffle=False)
        trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",
                                max_epochs=1000,
                                detect_anomaly=True)
        pdf = trainer.predict(saved_model, pred_data)

        return pdf
    else:

        trainset = LeakageData(L_train, s_train)
        traindata = DataLoader(trainset, batch_size=train_batch, num_workers=50, shuffle=True)
        early_stop_callback = _EarlyStopping(monitor="val_loss", patience=10, mode="min")
        pi_log_callback = PICallback()
        chkpt_callback = _ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", dirpath=chkpt_dir_path, filename=filename)
        # bar = LitProgressBar(f"{d_name} {len(L_train)}")
        # tb_logger = pl_loggers.TensorBoardLogger(save_dir="share_model", name=f"{d_name}_{model}_on_shares")
        trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",
                                max_epochs=1000,
                                check_val_every_n_epoch=2,
                                callbacks=[early_stop_callback, pi_log_callback, chkpt_callback],
                                detect_anomaly=True)
        trainer.fit(mlp, traindata, valdata)
        pi  = trainer.callbacks[1].PI
        pred_data = DataLoader(valset, batch_size=L_val.shape[0],num_workers=10, shuffle=False)
        pdf = trainer.predict(mlp, pred_data)
        return pdf, pi
def fix_s(s):
    s_ = s.copy().astype(np.int16)
    s_[s_==3328] = -1
    s_[s_==3327] = -2
    return s_
def NLLLoss(preds, targets):
    PRIOR = np.array([0.375, 0.25, 0.0625, 0.0625, 0.25])
    preds[np.where(~np.isfinite(preds))] = 1e-100
    preds[np.where(preds < 1e-100)] = 1e-100
    preds_log = np.log2(preds)
    # out = np.zeros_like(targets, dtype=np.float32)
    # for i in range(len(targets)):
    #     out[i] = preds_log[i][targets[i]]
    ce = 0
    # sec, sec_count = np.unique(targets, return_counts=True)
    for s_i in range(5):
        tmp = preds_log[targets==s_i, s_i]
        ce += tmp.mean()*PRIOR[s_i]
    return -ce

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
def mlp_sasca(leakage_handler, traces, labels, n_profiling, n_pois=10, model=None):
    total_n_traces = leakage_handler.n_files*leakage_handler.traces_per_file
    PoI = leakage_handler.PoI.reshape(leakage_handler.n_shares, n_pois)
    pdfs = []
    n_val = 500000
    train_idx, val_idx = under_sample(total_n_traces, n_profiling, labels, leakage_handler.n_shares, n_val=n_val)
    PI_SHARES = []
    secret_val = labels[val_idx, 2]
    secret_val = fix_s(secret_val)%5
    for i in trange(leakage_handler.n_shares, desc="PDFSHARE", leave=False):
        traces_ = traces[:, PoI[i]]
        Li_train = traces_[train_idx[i]]
        Li_val = traces_[val_idx]
        si_train = labels[train_idx[i], i]
        si_val = labels[val_idx, i]
        print(Li_train.shape)
        # res, pi = share_model(leakage_handler.d_name, Li_train, Li_val, si_train, si_val, share_i=i, load_saved=0, model=ID)
        res = share_model(leakage_handler.d_name, Li_train, Li_val, si_train, si_val, share_i=i, load_saved=1, model=ID)
        # print(np.array(pi).max())
        # PI_SHARES.append(pi)
        pdf_model = res[0].numpy(force=True)
        for i in range(1, len(res)):
            pdf_model = np.vstack((pdf_model, res[i].numpy(force=True)))
        if model is HW:
            pdf_share = pdf_model[:, HW(Zq)]
            pdf_share = pdf_share/ pdf_share.sum(axis=1, keepdims=True)
        # else:
        #     pdf_share = (pdf_model.T / np.sum(pdf_model,axis=1)).T
        pdfs.append(pdf_model.copy())

    # integrate to graph
    # graph_desc = gen_graph(n_shares=leakage_handler.n_shares)
    # graph = FactorGraph(graph_desc)
    # bp = BPState(graph, n_val)
    # for i in range(leakage_handler.n_shares):
    #     if leakage_handler.m_flag==10:
    #         bp.set_evidence(f"S{i}", pdfs[i].astype(np.float64))
    #     else:
    #         reverse_idx = (KYBER_Q - np.arange(KYBER_Q))%KYBER_Q
    #         pdf_si = pdfs[i] if i== leakage_handler.n_shares-1 else pdfs[i][:, reverse_idx]
    #         bp.set_evidence(f"S{i}", pdf_si.astype(np.float64))
    #
    # bp.bp_acyclic("S")
    # resS = bp.get_distribution("S")
    # pr_s = np.zeros((len(val_idx), 5))
    #
    # pr_s[:, range(5)] = resS[:, [0, 1, 2, -2, -1]]
    # pri_s = np.array([0.375, 0.25, 0.0625, 0.0625, 0.25])
    # pr_s = pr_s*pri_s
    # pr_s = pr_s/pr_s.sum(axis=1, keepdims=True)


    pr_s = sasca(pdfs, m_flag=0)
    pr_s[np.where(~np.isfinite(pr_s))] = 1e-100
    pr_s[np.where(pr_s < 1e-100)] = 1e-100
    entS = ent_s(prior_s)
    ce = CE_(pr_s, secret_val)
    return entS - ce, PI_SHARES


def pi_curve_mlp_wn(d_name, N_TRAIN, n_pois=10, model=ID):
    leakage_handler = Leakage_Handler(d_name, 195)
    leakage_handler.n_files = 400
    leakage_handler.get_PoI(n_pois=n_pois, mode="on_shares", model=model)
    n_shares = leakage_handler.n_shares
    total_n_traces = leakage_handler.n_files*leakage_handler.traces_per_file
    traces, labels_ = leakage_handler.get_data("full_trace")
    labels = labels_[:, [0, 1, 2]]
    print_centered("=========DATA LOAD==========")
    pi_holder = np.zeros(len(N_TRAIN))
    ns_pbar = tqdm(enumerate(N_TRAIN), total=len(N_TRAIN), leave=True)
    for i_n, n_traces in ns_pbar:
        ns_pbar.set_description_str(f"{d_name} N_TRACES: {n_traces}")
        pi, PI_SHARES = mlp_sasca(leakage_handler, traces, labels, n_traces, n_pois=n_pois, model=ID)
        ns_pbar.set_postfix_str(f"PI {pi:0.4f}")
        pi_holder[i_n] = pi
    return pi_holder
def exp_run(d_name):
    n_pois = 100
    N_TRAIN = [4000, 10000, 25000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, 750000, 1000000, 1500000, 1750000, 1800000]
    leakage_handler = Leakage_Handler(d_name, 195)
    leakage_handler.n_files = 200
    leakage_handler.get_PoI(n_pois=n_pois, mode="on_shares", model=ID)
    n_shares = leakage_handler.n_shares
    total_n_traces = leakage_handler.n_files*leakage_handler.traces_per_file
    traces, labels_ = leakage_handler.get_data("full_trace")
    labels = labels_[:, [0, 1]]
    ns_pbar = tqdm(enumerate(N_TRAIN), total=len(N_TRAIN), leave=True)
    pi_holder = np.zeros(len(N_TRAIN))
    f_log_buffer = f"log/pi_buffer_mlp_sasca_{d_name}_{n_pois}.npy"
    with open(f_log_buffer, "wb") as f_buff:
        np.save(f_buff, N_TRAIN)
    for i_n, n_traces in ns_pbar:
        ns_pbar.set_description_str(f"{d_name} N_TRACES: {n_traces}")
        if n_traces < 50000:
            pi = 0
            REPS = 5
            for i_rep in trange(REPS, desc="REPS"):
                pi_rep, PI_SHARES = mlp_sasca(leakage_handler, traces, labels, n_traces, n_pois=n_pois, model=ID)
                with open(f"log/pi_share_mlp_sasca_{d_name}_{n_traces}_{n_pois}.npy", "wb") as f:
                    for i_share in range(n_shares):
                        np.save(f, PI_SHARES[i_share])
                pi += pi_rep
            pi = pi/5
        else:
            pi, PI_SHARES = mlp_sasca(leakage_handler, traces, labels, n_traces, n_pois=n_pois, model=ID)
            with open(f"log/pi_share_mlp_sasca_{d_name}_{n_traces}_{n_pois}.npy", "wb") as f:
                for i_share in range(n_shares):
                    np.save(f, PI_SHARES[i_share])
        ns_pbar.set_postfix_str(f"PI {pi:0.4f}")
        pi_holder[i_n] = pi
        with open(f_log_buffer, "ab") as f_buff:
            np.save(f_buff, pi)
    with open(f"log/pi_curve_mlp_sasca_{d_name}.npy", "wb") as f:
        np.save(f, N_TRAIN)
        np.save(f, pi_holder)
    return pi_holder
def share_model_check(d_name):
    lh = Leakage_Handler(d_name, 195)
    lh.n_files = 200
    lh.get_PoI(n_pois=50, mode="on_sec", model=ID, keep=True)
    n_shares = lh.n_shares
    traces, labels_ = lh.get_data("full_trace")
    print(traces.shape)
    # traces = traces.astype(np.int16)
    labels = labels_[:, [0]].squeeze()
    N_PROFILING = [100000, 200000, 300000, 400000, 500000]
    for n_p in N_PROFILING:
        trace_train = traces[:n_p]
        label_train = labels[:n_p]
        print(np.random.choice(label_train, 10).tolist())
        trace_val = traces[-200000:]
        label_val = labels[-200000:]
        p_share, pi_share = share_model(d_name, trace_train, trace_val, label_train, label_val, model=ID)
        # p_share = p_share/ p_share.sum(axis=1, keepdims=True)
        print(len(p_share), p_share[0].shape)
        # pi = np.log2(KYBER_Q) - CE(p_share)
        plt.plot(range(len(pi_share)), pi_share, label=f"{n_p}")
        # print_centered(f"N_PROFILING: {n_p} PI: {pi}")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    N_PROFILING = [1500000, 2000000, 2500000]
    pi_curve = pi_curve_mlp_wn("280623_1200", N_PROFILING)
    plt.plot(N_PROFILING, pi_curve)
    pi_curve = pi_curve_mlp_wn("280623_1226", N_PROFILING)
    plt.plot(N_PROFILING, pi_curve)
    plt.show()
    # N_TRAIN = [4000, 10000, 25000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, 750000, 1000000, 1500000, 1600000, 1750000]
    #
    # pi = exp_run("280623_1200")
    # plt.plot(N_TRAIN, pi, label="sub")
    # pi = exp_run("280623_1226")
    # plt.plot(N_TRAIN, pi, label="add")
    # plt.show()
    # N_TRAIN = [800000, 900000, 1000000, 1250000]
    # pi = pi_curve_mlp_wn("280623_1200", N_TRAIN, n_pois=50, model=ID, mode="on_shares")
    # with open(f"mlp_shares_280623_1200_poi50.npy", "wb") as f:
    #     np.save(f, N_TRAIN)
    #     np.save(f, pi)
    # plt.plot(N_TRAIN, pi, label=f"S=X1+X2")
    # pi = pi_curve_mlp_wn("280623_1226", N_TRAIN, n_pois=50, model=ID, mode="on_shares")
    # with open(f"mlp_shares_280623_1226_poi50.npy", "wb") as f:
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
