import numpy as np
import os
s_range = np.array([-2, -1, 0, 1, 2])
prior_s = np.array([0.0625, 0.25,   0.375,  0.25,   0.0625])
KYBER_Q = 3329
HW_Q = 12
Zq = np.arange(KYBER_Q)
# width = 1
width = os.get_terminal_size().columns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend import Legend
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple

from tqdm import tqdm, trange
from scalib.metrics import SNR

def print_centered(str):
    print(str.center(width))
def count_1(x):
    return int(x).bit_count()
fcount = np.vectorize(count_1)
def HW(x):
    return fcount(x)
hw_set = np.unique(HW(Zq))

def ID(x):
    return x
def mean_traces(traces, labels):
    class_labels = np.unique(labels)
    n_classes = class_labels.shape[0]
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(traces[labels==cl], axis=0))
    return np.array(mean_vectors)
def pdf_normal(x, mu, sigma):
    ep = (x-mu)/sigma
    ep = -ep**2/2
    return np.exp(ep) / (sigma * np.sqrt(2*np.pi))
def under_sample(total_n_traces, n_traces, labels, n_shares, n_val=200000):
    train_idx = np.zeros((n_shares, n_traces), dtype=np.uint32)
    for share_i in range(n_shares):
        labels_i = labels[range(total_n_traces), share_i]
        idx_list = np.zeros(n_traces, dtype=np.uint32)
        for qi in range(KYBER_Q):
            clsq = np.where(labels_i == qi)[0]
            idx_list[qi] = np.random.choice(clsq, 1)
        avai_idx_ = np.setdiff1d(np.arange(total_n_traces), idx_list)
        idx_list[KYBER_Q: ] = np.random.choice(avai_idx_, n_traces-KYBER_Q, replace=False)
        train_idx[share_i] = idx_list.copy()
    avai_idx = np.setdiff1d(np.arange(total_n_traces), train_idx.flatten())
    if len(avai_idx) < n_val:
        val_idx = avai_idx
    else:
        val_idx = np.random.choice(avai_idx, n_val, replace=False)
    return train_idx, val_idx

def pdf_l_given_s(l, model=ID, sigma=0.1, q=KYBER_Q):
    id_range = np.arange(q)
    hw_range = np.unique(HW(id_range))
    val_range = id_range if model is ID else hw_range
    pdf = pdf_normal(l, val_range, sigma).copy()
    return pdf

def ent_s(pr=prior_s):
    """Entropy for prior proba
    """
    return np.nansum(-(np.log2(pr) * pr))

# def CE(resX):
#
#     return np.nansum(-(np.log2(resX) * resX), axis=1).mean()
def CE(resX):
    tmp = resX*np.log2(resX)
    ce = tmp.sum(axis=1)
    return -ce.mean()
def CE_(p_s_l, S):
    ce = 0
    n_classes = np.unique(S)
    p_s_l = np.log2(p_s_l)
    for i in n_classes:
        tmp = p_s_l[S==i]
        n_s = len(tmp)
        p_ = tmp[range(n_s), i]
        ce += (n_s/len(S))*(p_.mean())
    return -ce
def gen_graph(n_shares, q=KYBER_Q):
    graph_desc = f'''NC {q}\nVAR MULTI S\n'''
    prop = '''PROPERTY S ='''
    for i in range(n_shares):
        graph_desc += f"VAR MULTI S{i}\n" #add share
        prop +=  f" S{i}\n" if (i == n_shares-1) else f" S{i} +"
    graph_desc += prop
    return graph_desc
def gen_secrets(n_states, n_coeffs=1, s_set=s_range, p_s=prior_s):
    states = np.empty((n_states, n_coeffs), dtype=np.int32)
    for i in range(n_states):
        states[i] = np.random.choice(s_set, n_coeffs, p=p_s)
    return states

def gen_shares(secrets, n_shares=2, q=KYBER_Q, mode="sub"):
    shares = {}
    masked_state = secrets.copy()
    for i in range(n_shares-1):
        tmp = np.random.randint(0, q, size=secrets.shape, dtype=np.int32)
        if mode=="sub":
            masked_state = (masked_state-tmp)%q
        elif mode=="add":
            masked_state = (masked_state+tmp)%q
        elif mode=="add_one":
            masked_state = (masked_state+tmp)%q if i==0 else (masked_state-tmp)%q
        shares[f"S{i}"] = tmp.copy()
    shares[f"S{n_shares-1}"] = masked_state.copy()
    return shares

def gen_shares_ext(secrets, n_shares, sub_idx=0, q=KYBER_Q):
    if sub_idx != 0:
        sub_idx = np.array(sub_idx)

        assert len(sub_idx) <= n_shares - 1, f"SELECT INDEX OUT OF RANGE (#select<= {n_shares-1})"
        assert np.any(sub_idx <n_shares-1), f"SELECT INDEX OUT OF RANGE (<= {n_shares+1})"
    shares = {}
    masked_state = secrets.copy()%q
    for i in range(n_shares-1):
        tmp = np.random.randint(0, q, size=secrets.shape, dtype=np.int32)
        if isinstance(sub_idx, int):
            masked_state = (masked_state-tmp)%q
        else:
            masked_state = (masked_state+tmp)%q if i in sub_idx else (masked_state-tmp)%q
        shares[f"S{i}"] = tmp.copy()
    shares[f"S{n_shares-1}"] = masked_state.copy()
    return shares
def enc_repr(n_shares, sub_idx):
    if isinstance(sub_idx, int):
        ops = ["+" for i in range(n_shares)]
    else:
        ops = []
        for i in range(n_shares):
            ops.append("-" if i in sub_idx else "+")
    expression = "S="
    for i in range(n_shares):
        expression += f"{ops[i]}S{i}"
    return expression
def shares_sanity_check(shares, secrets, n_shares, sub_idx=0, q=KYBER_Q):
    if isinstance(sub_idx, int):
        ops = ["+" for i in range(n_shares)]
    else:
        ops = []
        for i in range(n_shares):
            ops.append("-" if i in sub_idx else "+")
    i_s = 0
    expression = "S="
    tmp = 0
    for share, share_vals in shares.items():
        print(f"{share}")
        print(f"{share_vals.tolist()}")
        if isinstance(sub_idx, int):
            tmp = (tmp + share_vals)%q
        else:
            tmp = (tmp - share_vals)%q if i_s in sub_idx else (tmp + share_vals)%q
            print(i_s, i_s in sub_idx, share, tmp.tolist())
        expression += f"{ops[i_s]}{share}"
        i_s += 1
    print(expression)
    tmp = tmp.squeeze()%q
    s_tmp= secrets.copy()
    s_tmp = s_tmp.squeeze()%q
    print(tmp.tolist(), s_tmp.tolist())
    print("CORRECT" if np.array_equal((tmp.squeeze())%q, s_tmp%q) else "INCORRECT")





def gen_leakages(shares, sigma, model):
    L = {}
    for share, share_val in shares.items():
        L[share] = model(share_val) + np.random.normal(0, sigma, size=share_val.shape)
    return L
def gen_sigma(n_shares):
    log_s2_small = np.array([-3, -2.5, -2, -1.75])
    n_samples = np.ones_like(log_s2_small, dtype=np.uint32)*(n_shares+3)*10000
    log_s2_large = np.append(np.linspace(-1.5, -0.5, 10, endpoint=False), np.linspace(-0.5, 2, 8, endpoint=True))
    s_snr = 4/np.array([10, 1, 0.1])
    s_snr = np.log10(s_snr)
    idx = np.searchsorted(log_s2_large, s_snr)
    log_s2_large = np.insert(log_s2_large, idx, s_snr)
    n_samples = np.append(n_samples, (n_shares+3)*10000*((log_s2_large + 5).astype(np.int32)))
    log_s2 = np.append(log_s2_small, log_s2_large)
    sigmas = np.power(10, log_s2)
    sigmas = np.sqrt(sigmas)
    return sigmas, n_samples, log_s2
# with open(f"log/pi_sasca_3329_sub_3shares.npy", "rb") as f:
#     for i in range(5):
#         print(np.load(f))
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle

def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    """
    Place a table legend on the axes.

    Creates a legend where the labels are not directly placed with the artists,
    but are used as row and column headers, looking like this:

    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |


    Parameters
    ----------

    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.

    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.

    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.

    title_label : str, optional
        Label for the top left corner in the legend table.

    ncol : int
        Number of columns.


    Other Parameters
    ----------------

    Refer to `matplotlib.legend.Legend` for other parameters.

    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')

    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]

        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]

        # empty label
        empty = [""]

        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol

        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow
        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, fontsize=10, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        for line in ax.legend_.get_lines():
            line.set_linewidth(1.5)
        return ax.legend_
enc_mode = ["SUB", "ADD"]
def get_info_from_log(dir_name):
    f_log = f"{dir_name}/log"
    info = {}
    keys = ["n_files", "poly_per_batch", "n_batchs", "n_shares", "m_flag", "fname", "n_samples"]
    with open(f_log, "r") as f:
        f_ls = f.readlines()
    for line in f_ls:
        for key in keys:
            if key in line:
                val = line.rstrip().split("=")[1] if key=="fname" else int( line.rstrip().split("=")[1])
                info[key] = val
                break
    return info
class Leakage_Handler:

    def __init__(self, d_name, n_samples):
        self.d_name = d_name
        if os.path.isdir(f"traces/{d_name}"):
            self.dir_name = f"traces/{d_name}"
        else:
            self.dir_name = f"/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415_fastA/Measurements/traces/{d_name}"
        info = get_info_from_log(self.dir_name)
        self.n_shares = info["n_shares"]
        self.n_samples = n_samples*self.n_shares
        self.n_files = info["n_files"]
        self.traces_per_file = info["poly_per_batch"]*info["n_batchs"]
        self.m_flag = info["m_flag"]
        self.fname = info["fname"]
        self.file_temp = f"{self.dir_name}/{self.fname}_{self.n_shares}"

    def get_PoI(self, n_pois, mode, model, keep=False, display=False):
        self.n_pois = n_pois
        if mode=="on_shares":
            if model is HW:
                snr = SNR(HW_Q, self.n_samples, self.n_shares, use_64bit=True)
            else:
                snr = SNR(model(KYBER_Q), self.n_samples, self.n_shares, use_64bit=True)
        elif mode=="on_sec":
            if model is ID:
                snr = SNR(KYBER_Q, self.n_samples, 1, use_64bit=True)
            else:
                snr = SNR(HW_Q, self.n_samples, 1, use_64bit=True)
        else:
            self.n_pois = self.n_samples
            self.PoI = np.arange(self.n_samples)
            return None
        for fi in trange(self.n_files, desc="SNR|File"):
            f_t = f"{self.file_temp}_traces_{fi}.npy"
            f_d = f"{self.file_temp}_meta_{fi}.npz"
            traces = np.load(f_t).astype(np.int16)
            data = np.load(f_d)["polynoms"]
            if mode=="on_shares":
                polys = data[range(self.traces_per_file), :-1].astype(np.uint16)
            elif mode=="on_sec":
                polys = data[range(self.traces_per_file), -1:].astype(np.uint16)
            polys = model(polys).astype(np.uint16)
            snr.fit_u(traces, polys)
        snr_val = snr.get_snr()

        if mode=="on_sec":
            self.PoI = np.argsort(snr_val[0])[-n_pois:]
        elif mode=="on_shares":
            self.PoI = np.zeros((self.n_pois*self.n_shares,), dtype=np.int16)
            for c in range(self.n_shares):
                idx = np.argsort(snr_val[c])[-n_pois:]
                self.PoI[c*self.n_pois: c*self.n_pois+self.n_pois] = idx
        if display:
            if mode=="on_sec":
                plt.plot(snr_val.T, label="secret")
            elif mode=="on_shares":
                for c in range(self.n_shares):
                    plt.plot(range(self.n_samples), snr_val[c], label=f"share {c+1}")
            plt.title(f"SNR {enc_mode[self.m_flag%10]}")
            plt.legend()
            plt.show()
        if keep:
            self.snr = snr_val
        return None
    def traces_trim(self, mode):
        fnew = f"{self.file_temp}_traces_0_seconly.npy" if mode=="on_sec" else f"{self.file_temp}_{self.n_pois}traces_0.npy"
        if os.path.exists(fnew):
            print_centered("DATA IS READY!")
            pass
        else:
            for fi in trange(self.n_files, desc="Trimming|FILE"):
                f_t = f"{self.file_temp}_traces_{fi}.npy"
                traces = np.load(f_t).astype(np.int16)
                fnew = f"{self.file_temp}_traces_{fi}_seconly.npy" if mode=="on_sec"  else f"{self.file_temp}_{self.n_pois}traces_{fi}.npy"
                with open(fnew, "wb") as f:
                    np.save(f, traces[:, self.PoI].copy())

    def get_data(self, mode, keep=False):
        if mode=="full_trace":
            L = np.zeros((self.traces_per_file*self.n_files, self.n_samples))
        elif mode=="on_sec":
            L = np.zeros((self.traces_per_file*self.n_files, self.n_pois))
        elif mode=="on_shares":
            L = np.zeros((self.traces_per_file*self.n_files, self.n_pois*self.n_shares))
        elif mode=="on_shares_sep":
            #NOTE: fix this back after being done Analysis
            L = np.zeros((self.n_shares, self.traces_per_file*self.n_files, self.n_pois))
            # L = np.zeros((self.n_shares, self.traces_per_file*self.n_files, self.n_samples))
        labels = np.zeros((self.traces_per_file*self.n_files, self.n_shares+1))
        if mode != "full_trace":
            self.traces_trim(mode)
        for fi in trange(self.n_files, desc="GETTING DATA|FILE"):
            i_f = fi*self.traces_per_file
            if mode in ["full_trace", "on_shares_sep"] :
                f_t = f"{self.file_temp}_traces_{fi}.npy"
            else:
                if mode=="on_sec":
                    f_t = f"{self.file_temp}_traces_{fi}_seconly.npy"
                elif mode=="on_shares":
                    f_t = f"{self.file_temp}_{self.n_pois}traces_{fi}.npy"
            f_d = f"{self.file_temp}_meta_{fi}.npz"
            traces = np.load(f_t).astype(np.int16)
            data = np.load(f_d)["polynoms"]
            labels[i_f: i_f+self.traces_per_file] = data[range(self.traces_per_file), :]
            if mode != "on_shares_sep":
                L[i_f: i_f+self.traces_per_file] = traces
            else:
                for share_i in range(self.n_shares):
                    self.n_samples
                    idx = share_i*self.n_pois
                    # L[share_i, i_f: i_f+self.traces_per_file] = traces[:, self.PoI[idx:idx+self.n_pois]]
                    L[share_i, i_f: i_f+self.traces_per_file] = traces
                    # labels[share_i, i_f: i_f+self.traces_per_file] = data[range(self.traces_per_file), share_i]
        if keep:
            self.traces =  L
            self.labels = labels
            return 0
        else:
            return L.astype(np.float32), labels
class GT:
	""" Used to compute Gaussian Templates based on
		Observations"""

	def __init__(self,Nk=256,Nd=1):

		self._Nks = np.zeros(Nk,dtype=np.float64)
		self._sums = np.zeros((Nk,Nd),dtype=np.float64)
		self._muls = np.zeros((Nk,Nd,Nd),dtype=np.float64)
		self._Nk = Nk
		self._Nd = Nd

	def fit(self,traces,keys):
		traces = traces[:,:self._Nd]
		N = np.zeros(self._Nk)
		sums = np.zeros((self._Nk,self._Nd))
		mults = np.zeros((self._Nk,self._Nd,self._Nd))

		for k in range(self._Nk):
			indexes = np.where(keys==k)[0]
			self._Nks[k] += len(indexes)
			self._sums[k,:] += np.sum(traces[indexes,:],axis=0)
			self._muls[k,:] += (np.dot(traces[indexes,:].T,traces[indexes,:]))

	def merge(self,gt):
		self._sums += gt._sums
		self._muls += gt._muls
		self._Nks += gt._Nks

	def get_template(self):
		means = np.zeros((self._Nk,self._Nd))
		vars = np.zeros((self._Nk,self._Nd,self._Nd))

		for k in range(self._Nk):
			u = self._sums[k]/self._Nks[k]

			N = self._Nd
			var = (self._muls[k,:]/self._Nks[k]) - (np.tile(u,(N,1)).T*u).T

			means[k,:] = u
			vars[k,:,:] = var

		return means,vars


	def input_dist(self):
		return self._Nks/np.sum(self._Nks)
