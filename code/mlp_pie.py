from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar
# from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning import Callback
from lightning.pytorch import loggers as pl_loggers
from sklearn.model_selection import train_test_split
from helpers import *
from scalib.metrics import SNR
from mlp_arch import *
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

torch.autograd.set_detect_anomaly(True)

class GetGrad(Callback):
    def __init__(self):
        super().__init__()
        self.input_grads = []
#
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.input_grads.append(pl_module.grad)


class GetBNWeights(Callback):
    def __init__(self):
        super().__init__()
        self.bnweights = []
#
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.bnweights.append(pl_module.bnweights)
#


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

def fix_s(s):
    s_ = s.copy().astype(np.int16)
    s_[s_==3328] = -1
    s_[s_==3327] = -2
    return s_

def mlp_pi(d_name, mode, n_pois, model=ID):
    print_centered(f"================MLP PROCESS {d_name} {mode} {model}=================")
    leakage_handler = Leakage_Handler(d_name, n_samples=195)
    leakage_handler.get_PoI(n_pois, mode, model)
    traces, data = leakage_handler.get_data(mode)
    labels = data[range(len(data)), -1]
    labels = fix_s(labels)%5
    # traces = traces.astype(np.float32)
    L_train, L_val, s_train, s_val = train_test_split(traces, labels, test_size=0.2)
    train_batch = len(L_train)//50
    val_batch = len(L_val)//20

    trainset = LeakageData(L_train, s_train)
    traindata = DataLoader(trainset, batch_size=train_batch, num_workers=10, shuffle=True)

    valset = LeakageData(L_val, s_val)
    valdata = DataLoader(valset, batch_size=val_batch,num_workers=10, shuffle=False)
    mlp = MLP_SEC(traces.shape[-1])
    early_stop_callback = _EarlyStopping(monitor="val_loss", patience=50, mode="min")
    pi_log_callback = PICallback()
    get_grad = GetGrad()
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs", name=f"{d_name}_{mode}")
    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",logger=tb_logger, max_epochs=500, check_val_every_n_epoch=1,
    callbacks=[early_stop_callback, pi_log_callback, get_grad], detect_anomaly=True)
    trainer.fit(mlp, traindata, valdata)
    pi  = trainer.callbacks[1].PI
    grads = trainer.callbacks[2].input_grads
    return np.array(pi), np.array(grads)

def pi_est(d_name, L_train, L_val, s_train, s_val , log_decs):
    train_batch = len(L_train)//50 if len(L_train) > 1000 else len(L_train)//5
    val_batch = len(L_val)//20 if len(L_val) > 1000 else len(L_val)//2

    n_w = 10  if len(L_val) <= 1000 else 20

    trainset = LeakageData(L_train, s_train)
    traindata = DataLoader(trainset, batch_size=train_batch, num_workers=n_w, shuffle=True)

    valset = LeakageData(L_val, s_val)
    valdata = DataLoader(valset, batch_size=val_batch,num_workers=n_w, shuffle=False)
    mlp = MLP_SEC(L_train.shape[-1])
    early_stop_callback = _EarlyStopping(monitor="val_loss", patience=100, mode="min")
    pi_log_callback = PICallback()
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir="mlp_pie", name=f"{d_name}_{log_decs}")
    # trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",logger=tb_logger, max_epochs=700, check_val_every_n_epoch=1,
    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto", max_epochs=700, check_val_every_n_epoch=1,
    callbacks=[early_stop_callback, pi_log_callback], detect_anomaly=True)
    trainer.fit(mlp, traindata, valdata)
    pi  = trainer.callbacks[1].PI
    return np.array(pi)
def pi_curve_n_samples(d_name, mode, n_pois, model=ID):
    model_desc = "HW" if model is HW else "ID"
    leakage_handler = Leakage_Handler(d_name, n_samples=195)
    total_n_traces = leakage_handler.n_files*leakage_handler.traces_per_file
    if mode == "on_shares":
        leakage_handler.get_PoI(n_pois, mode, model)
        traces, data = leakage_handler.get_data(mode)
    labels = data[range(len(data)), -1]%5
    N_TRAIN = [ 10000, 50000, 100000, 250000, 300000, 400000]
    N_VAL = [ 100000, 250000, 250000, 250000, 100000, 100000]
    # N_TRAIN = [100, 500, 1000, 5000, 10000, 50000, 100000, 250000, 300000, 400000]
    # N_VAL = [10000, 10000, 10000, 50000, 100000, 250000, 250000, 250000, 100000, 100000]
    n_reps = 2
    ns_pbar = tqdm(enumerate(N_TRAIN), total=len(N_TRAIN))
    pi_holder = []
    for i_n, n_traces in ns_pbar:
        print_centered(f"================MLP PROCESS {d_name} {mode} {model_desc} {n_pois} {n_traces}=================")
        ns_pbar.set_description_str(f"N_TRACES: {n_traces}")
        log_decs = f"{d_name}_{model_desc}_{mode}_{n_pois}_{n_traces}"
        pi_avg = 0
        for rep in trange(n_reps, desc="REPS|"):
            training_traces = np.random.choice(np.arange(total_n_traces), n_traces, replace=False)
            avai_idx = np.setdiff1d(np.arange(total_n_traces), training_traces)
            val_traces = np.random.choice(avai_idx, N_VAL[i_n])
            L_train = traces[training_traces].copy()
            S_train = labels[training_traces].copy()
            L_val = traces[val_traces].copy()
            S_val = labels[val_traces].copy()
            pi = pi_est(d_name, L_train, L_val, S_train, S_val , log_decs)
            print(pi.max())
            pi_avg += pi.max()
        pi_avg /= n_reps
        pi_holder.append(pi_avg)
    plt.plot(pi_holder, label=f"{d_name}")
            # with open(f"log/mlp_pie_{d_name}_{mode}_{n_pois}_{n_traces}.npy", "ab") as f:
            #     np.save(f, pi)
def pi_validation_curve():
    d_names = ["280623_1200", "280623_1226"]
    m_flags = ["SUB", "ADD"]
    mlp_modes = ["full_trace", "on_sec", "on_shares"]
    for i, d_name in enumerate(d_names):
        for mlp_mode in mlp_modes:
            n_points = 50 if mlp_mode=="on_sec" else 10
            pi, grads = mlp_pi(d_name, mlp_mode, n_points, model=ID)
            PI = pi[~np.isnan(pi)]
            plt.plot(range(len(PI)), PI, label=f"{mlp_mode}")
            with open(f"log/mlp_{d_name}_{mlp_mode}_pi&grads.npy", "wb") as f:
                np.save(f, PI)
                np.save(f, grads)
            plt.legend()
            plt.title(f"{m_flags[i]}_Validation PI")
            plt.show()

if __name__ == '__main__':
    # pi_curve_n_samples("280623_1200", mode="on_shares", n_pois=10)
    # pi_curve_n_samples("280623_1226", mode="on_shares", n_pois=10)
    # pi_curve_n_samples("280623_1200", mode="on_shares", n_pois=50)
    pi_curve_n_samples("280623_1226", mode="on_shares", n_pois=50)
    plt.legend()
    plt.show()
    # with open("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/log/mlp_190623_1600_full_trace_pi&grads.npy", "rb") as f:
    #     pi = np.load(f)
    #     grads = np.load(f)
    #     print(grads.shape)
    # plt.plot(grads[-1:].T)
    # plt.show()
    # exit()
    # d_names = ["190623_1600"]
    # m_flags = ["SUB"]
    # mlp_modes = ["full_trace"]
    # for i, d_name in enumerate(d_names):
    #     for mlp_mode in mlp_modes:
    #         n_points = 50 if mlp_mode=="on_sec" else 10
    #         pi, grads = mlp_pi(d_name, mlp_mode, n_points, model=VAL)
    #         PI = pi[~np.isnan(pi)]
    #         plt.plot(range(len(PI)), PI, label=f"{mlp_mode}")
    #         with open(f"log/mlp_{d_name}_{mlp_mode}_pi&grads.npy", "wb") as f:
    #             np.save(f, PI)
    #             np.save(f, grads)
    #         plt.legend()
    #         plt.title(f"{m_flags[i]}_Validation PI")
    #         plt.show()
    # modes = ["SUB"]
    # mlp_modes = ["on_shares"]
    # d_names = ["260623_1210"]
    # for i, dn in enumerate(d_names):
    #     fn = f"mlp_{dn}_{modes[i]}_1poi.npy"
    #     for mlp_mode in mlp_modes:
    #         centered_pr(f"=============PROCESS MODE {modes[i]} {mlp_mode}=================")
    #         pi, grads = mlp_pi(dn, mode=mlp_mode)
    #         PI = pi[~np.isnan(pi)]
    #         print(modes[i], mlp_mode, PI)
    #         with open(fn, "ab") as f:
    #             np.save(f, PI)
    #             np.save(f, grads)
    #         centered_pr(f"=============PROCESS MODE {modes[i]} {mlp_mode} SAVED PI SAVED GRADS=================")

    # plt.legend()
    # plt.show()
    # d_names = ["190623_1600", "190623_1956"]
    # fn = "mlp_onshares_2shares.npy"
    # with open(fn, "wb") as f:
    #     np.save(f, np.array([10, 11]))
    #     for i, dn in enumerate(d_names):
    #         print(f"=============PROCESS MODE {modes[i]}=================")
    #         pi = mlp_pi(dn, mode="on_shares")
    #         PI = pi[~np.isnan(pi)]
    #         plt.plot(np.arange(len(PI)), PI, label=f"{modes[i]}")
    #         print(modes[i], PI)
    #         with open(fn, "ab") as f:
    #             np.save(f, PI)
    #
    # plt.legend()
    # plt.show()


    # PI_a = []
    # event_acc = EventAccumulator("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/lightning_logs/lightning_logs_server/version_15/events.out.tfevents.1687198756.Morgane.1340793.0")
    # event_acc.Reload()
    # log_event = event_acc.Scalars("PI")
    # for event in log_event:
    #     PI_a.append(event.value)
    # plt.plot(np.arange(len(PI_a)), PI_a, label="ADD")
    # plt.axvline(len(PI_a)-5, color="red", linestyle="dashed", alpha=0.5)
    # PI = []
    # event_acc = EventAccumulator("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/lightning_logs/lightning_logs_server/version_16/events.out.tfevents.1687199040.Morgane.1340793.1")
    # event_acc.Reload()
    # log_event = event_acc.Scalars("PI")
    # for event in log_event:
    #     PI.append(event.value)
    # plt.text(len(PI_a)-20, PI_a[-2]+0.01, f"{PI_a[-2]:0.4f}", fontsize=8, color="tab:blue")
    # plt.text(len(PI_a)-20, PI[len(PI_a)]+0.01, f"{PI[len(PI_a)]:0.4f}", fontsize=8, color="tab:orange")
    # plt.plot(np.arange(len(PI)), PI, label="SUB")
    # plt.title("Validation PI")
    # plt.legend()
    # plt.show()
    # print(PI_a[-2]/PI[-2])
