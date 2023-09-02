from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Callback
from lightning.pytorch import loggers as pl_loggers
from sklearn.model_selection import train_test_split
from helpers import *
from scalib.metrics import SNR
from mlp_arch import *
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scalib.attacks import SASCAGraph, FactorGraph, BPState
import gc
from time import time
from tqdm import trange
HW_Q = 12

torch.autograd.set_detect_anomaly(True)

class PICallback(Callback):
    """PyTorch Lightning PI callback."""

    def __init__(self):
        super().__init__()
        self.PI = []

    def on_validation_end(self, trainer, pl_module):
        self.PI.append(trainer.logged_metrics["PI"].item())

class _EarlyStopping(EarlyStopping, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
def share_model(d_name, traces, labels, model=None):
    train_val_split = len(traces)//5
    L_train = traces[:-train_val_split, ...]
    L_val = traces[-train_val_split: , ...]
    s_train = labels[:-train_val_split, ...]
    s_val = labels[-train_val_split: , ...]
    # L_train, L_val, s_train, s_val = train_test_split(traces, labels, test_size=0.25, random_state=42, shuffle=None)
    train_batch = L_train.shape[0]//50
    val_batch = L_val.shape[0]//20
    trainset = LeakageData(L_train, s_train)
    traindata = DataLoader(trainset, batch_size=train_batch,num_workers=10, shuffle=True)

    valset = LeakageData(L_val, s_val)
    valdata = DataLoader(valset, batch_size=val_batch,num_workers=10, shuffle=False)
    if model is None:
        mlp = MLP(in_dim=L_train.shape[-1], out_dim=KYBER_Q)
    else:
        mlp = MLP(in_dim=L_train.shape[-1], out_dim=HW_Q)

    early_stop_callback = _EarlyStopping(monitor="val_loss", patience=50, mode="min")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="share_model", name=f"{d_name}_{model}")
    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",
                            logger=tb_logger, max_epochs=20,
                            check_val_every_n_epoch=1, callbacks=[early_stop_callback], detect_anomaly=True)
    trainer.fit(mlp, traindata, valdata)
    pred_data = DataLoader(valset, batch_size=L_val.shape[0]//2,num_workers=10, shuffle=False)
    pdf = trainer.predict(mlp, pred_data)
    return pdf
def gen_graph(n_shares, q=KYBER_Q):
    graph_desc = f'''NC {q}\nVAR MULTI S\n'''
    prop = '''PROPERTY S ='''
    for i in range(n_shares):
        graph_desc += f"VAR MULTI S{i}\n" #add share
        prop +=  f" S{i}\n" if (i == n_shares-1) else f" S{i} +"
    graph_desc += prop
    return graph_desc
def get_share_data(L, data, n_points, share_id):
    Li = L[range(len(L)), share_id*n_points: share_id*n_points+n_points].copy()
    labelsi = data[range(len(L)), share_id].copy()
    return Li, labelsi
def CE(resX):
    return np.nansum(-(np.log2(resX) * resX), axis=1).mean()

def mlp_sasca(d_name, model=None):
    centered_pr(f"=============PROCESS {d_name} {model}==================")
    hw_q = HW(np.arange(KYBER_Q))
    #get share pdf
    n_points = 10
    info = get_info_from_log(d_name)
    m_flag = info["m_flag"]
    n_shares = info["n_shares"]

    if n_shares==3:
        L_, data_ = gen_data(d_name, n_points)
        L = L_[:1000000, :].copy()
        data = data_[:1000000, :].copy()
    else:
        L, data = gen_data(d_name, n_points)
    L = L.astype(np.float32)

    pdfs = []
    for i in trange(n_shares, desc="PDF share"):
        share_traces, share_labels = get_share_data(L, data, n_points, i)
        if model is not None:
            share_labels = HW(share_labels)
        res = share_model(d_name, share_traces, share_labels, model=model)
        pdf_model = res[0].numpy(force=True)
        for i in range(1, len(res)):
            pdf_model = np.vstack((pdf_model, res[i].numpy(force=True)))
        if model is not None:
            pdf_share = pdf_model[:, hw_q]
            pdf_share = pdf_share/ pdf_share.sum(axis=1, keepdims=True)
        else:
            pdf_share = (pdf_model.T / np.sum(pdf_model,axis=1)).T
        pdfs.append(pdf_share.copy())



    # pdf second share
    del share_traces
    del share_labels
    del pdf_model
    # del pdf_share
    gc.collect()
    # integrate to graph
    n_profiling = int(len(L)//5)
    graph_desc = gen_graph(n_shares=n_shares)
    centered_pr(graph_desc)
    centered_pr(f"===========MLP SASCA PROCESS {d_name} Model {model} MODE: {m_flag}==============")
    graph = FactorGraph(graph_desc)
    bp = BPState(graph, n_profiling)
    for i in range(n_shares):
        if m_flag==10:
            bp.set_evidence(f"S{i}", pdfs[i].astype(np.float64))
        else:
            reverse_idx = (KYBER_Q - np.arange(KYBER_Q))%KYBER_Q
            pdf_si = pdfs[i] if i== n_shares-1 else pdfs[i][:, reverse_idx]
            bp.set_evidence(f"S{i}", pdf_si.astype(np.float64))

    st = time()
    bp.bp_acyclic("S")
    print("SASCA solve", time()-st)
    s_range = np.array([-2, -1, 0, 1, 2])
    resS = bp.get_distribution("S")
    pr_s = np.zeros((n_profiling, 5))
    pr_s[:, s_range] = resS[:, s_range]
    prior_S = np.array([0.375,0.25,0.0625,0.0625,0.25])
    pr_s *= prior_S
    pr_s = (pr_s.T / np.sum(pr_s,axis=1)).T
    entS = ent_s(prior_S)
    ce = CE(pr_s)
    print(entS-ce)
    return entS - ce
if __name__ == '__main__':
    # x = np.array([1, 2, 3328, 3327, 0])
    # y = (x - 3329)
    # print(y)
    # pi_sub = mlp_sasca("190623_1600")
    # pi_add = mlp_sasca("200623_2115")
    # centered_pr("===============MLPSASCA 2 shares share_val model==============")
    # centered_pr(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    # centered_pr("==================================")
    # pi_sub = mlp_sasca("200623_1211")
    # pi_add = mlp_sasca("200623_1225")
    # centered_pr("===============MLPSASCA 3 shares share_val model==============")
    # centered_pr(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    # centered_pr("==================================")
    pi_sub = mlp_sasca("190623_1600", model="HW")
    pi_add = mlp_sasca("200623_2115", model="HW")
    centered_pr("===============MLPSASCA 2 shares HW model==============")
    centered_pr(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    centered_pr("==================================")
    # pi_sub = mlp_sasca("200623_1211", model="HW")
    # pi_add = mlp_sasca("200623_1225", model="HW")
    # centered_pr("===============MLPSASCA 3 shares HW model==============")
    # centered_pr(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    # centered_pr("==================================")

    # y_leakage = np.random.rand(2, 256) # this might come from an LDA
    # print(y_leakage.shape)
    # y_leakage = y_leakage / y_leakage.sum(axis=1, keepdims=True)
    # print(y_leakage.shape, y_leakage.sum(axis=1))

    # share_model("190623_1600", 10, 0)
