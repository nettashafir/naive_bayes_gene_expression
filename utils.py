import os
import sys
import numpy as np
from functools import reduce
from itertools import permutations
import pandas as pd
from tqdm import tqdm
from typing import Callable, Optional
import pytz
import csv
import time
from datetime import datetime
from scipy.stats import nbinom, multivariate_normal as mvnorm
from scipy.special import gamma, loggamma, digamma, polygamma, logsumexp as LSE
from scipy.optimize import root, brentq, linear_sum_assignment, fsolve, newton, minimize, RootResults, OptimizeResult
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# simulated data parameters
M = 5000  # samples
N = 10  # dimension of every sample
K = 5  # hidden states (support of C)
D = 20  # In the case of categorical CPD - the maximum value

# EM parameters
THETA_C_SMOOTHING_EPSILON = None  # 1e-7
LOG_POSTERIOR_SMOOTHING_EPSILON = None  # 1e-1
STOP_ITERATION_EPSILON = 1e-7
N_ROOT_FINDER_INITIALIZATIONS = 7
DEF_MAX_ITER = 100
DEF_SEED = 42
DEF_SEED_GENERATE_DATA = 42
HEAD_EVALUATE = 5

# CPD's distribution enum
CATEGORICAL_EM_CODE = 0
NEGATIVE_BINOMIAL_EM_CODE = 1
NEGATIVE_BINOMIAL_SEQ_DEPTH_EM_CODE = 2

# main parameters
REAL_DATA = False
DATA_PATH = None
CPD_CODE = NEGATIVE_BINOMIAL_SEQ_DEPTH_EM_CODE
SAVE_RESULTS = False
RANDOM_NOISE_INIT = True
DEBUG = False

# feature selection parameters
OVER_EXPRESSED_ONLY = True
ABSOLUTE_GAP_FROM_SECOND = 7
ABSOLUTE_GAP_FROM_THIRD = 3
VAR_THRESHOLD = 10
FOLD_CHANGE_THRESHOLD = 2

# locations
DATA_LOCATION = "data/"   # FILL HERE IF RUNNING ON REAL DATA


# ---------------------------- EM utils

def calc_log_marginal_and_posterior(calc_logpmf_mat, data, theta_C, theta_X, **kwargs):
    """
      TODO: document, use shape_error
    """
    M, N = data.shape
    K = theta_C.shape[0]

    log_pmf = calc_logpmf_mat(data, theta_C, theta_X, **kwargs)   # shape: (K, M), log_pmf[k, m] = log(p(x[m] | C=k))
    log_likelihood = log_pmf + np.log(theta_C[:, None])           # (K, M), log_likelihood[k, m] = log(p(x[m], C=k))
    log_marginal = LSE(log_likelihood, axis=0)                    # (M, ), log_marginal[m] = log(p(x[m]))
    log_posterior = log_likelihood - log_marginal                 # (K, M), log_posterior[k, m] = log(p(C=k | x[m]))

    # Smooth Posterior
    ESS_C = np.exp(LSE(log_posterior, axis=1))
    if LOG_POSTERIOR_SMOOTHING_EPSILON is not None and np.any(ESS_C < 1):
        if DEBUG:
            print("SMOOTHING LOG_POSTERIOR!")
        log_epsilon_matrix = np.full((K, M), np.log(LOG_POSTERIOR_SMOOTHING_EPSILON))
        numerator = LSE(np.concatenate((log_posterior[None, :], log_epsilon_matrix[None, :]), axis=0), axis=0)
        normalization = LSE(numerator, axis=0)
        log_posterior = numerator - normalization

    # Sanity checks
    if np.any(np.isnan(log_posterior)):
        raise ValueError("log posterior has nan entries - there is a state with 0 probability")
    assert log_marginal.shape[0] == M, f"log marginal vector shape is {log_marginal.shape}, should be ({M},)"
    assert log_posterior.shape == (K, M), f"log posterior matrix shape is {log_posterior.shape}, should be ({K}, {M})"
    assert np.all(np.isclose(LSE(log_posterior, axis=0), 0)), f"Not all the posteriors are summed to 1: {LSE(log_posterior, axis=0)[:100]}"

    return log_marginal, log_posterior


def calc_normalized_log_marginal_likelihood(calc_logpmf_mat, data, theta_C, theta_X, **kwargs):
    M, N = data.shape
    log_marginal, _ = calc_log_marginal_and_posterior(calc_logpmf_mat, data, theta_C, theta_X, **kwargs)
    normalized_log_marginal = np.sum(log_marginal) / (M * N)
    return normalized_log_marginal


def calc_log_ESS_count(data, log_posteriors_matrix):
    M, N = data.shape
    K, _ = log_posteriors_matrix.shape
    D = int(np.max(data)) + 1

    matrix_index = np.repeat(np.arange(N)[None, :], M, axis=0).T
    matrix_index_tensor = np.repeat(matrix_index[:, None], K, axis=1)
    rows_index = np.repeat(np.arange(K)[None, :], M, axis=0).T
    rows_index_tensor = np.repeat(rows_index[None, :], N, axis=0)
    col_index_tensor = np.repeat(data.T[:, None], K, axis=1).astype(int)
    log_posterior_tensor = np.repeat(log_posteriors_matrix[None, :], N, axis=0)

    log_ESS_count = np.zeros((N, K, D))
    b = np.max(log_posterior_tensor)
    np.add.at(log_ESS_count, [matrix_index_tensor, rows_index_tensor, col_index_tensor], np.exp(log_posterior_tensor - b))
    log_ESS_count = np.log(log_ESS_count)
    log_ESS_count += b

    return log_ESS_count


# ---------------------------- simulated data utils

def generate_categorical_data(rng, max_val=D, num_samples=M, num_variables=N, num_states=K, ):
    theta_C = rng.uniform(1, 10, num_states)
    theta_C = theta_C / theta_C.sum()
    theta_X = rng.dirichlet(np.ones(max_val), (num_variables, num_states))
    data = np.zeros((num_samples, num_variables))
    states = np.zeros(num_samples)
    for m in range(M):
        k = rng.choice(np.arange(K), 1, p=theta_C)[0]
        states[m] = k
        param = theta_X[:, k, :]
        data[m] = np.array([rng.choice(np.arange(max_val), 1, p=p)[0] for p in param])
    return theta_C, theta_X, data, states


def generate_data_NegativeBinomial(rng, r_max=100,
                                   num_samples=M, num_variables=N, num_states=K,
                                   random_theta_C=True, seq_depth=None):
    if seq_depth is None:
        seq_depth = rng.uniform(1, 10, num_samples)

    if random_theta_C:
        theta_C = rng.uniform(1, 10, num_states)
        theta_C = theta_C / theta_C.sum()
    else:
        theta_C = np.arange(num_states) + 1
        theta_C = theta_C / theta_C.sum()

    r = rng.uniform(low=1, high=r_max, size=(num_variables, num_states))
    p = rng.uniform(0, 1, (num_variables, num_states))

    data = np.zeros((num_samples, num_variables))
    states = np.zeros(num_samples)
    for m in range(num_samples):
        p_m = p / (p + seq_depth[m] - seq_depth[m] * p)
        k = rng.choice(np.arange(num_states), p=theta_C)
        states[m] = k
        data[m] = rng.negative_binomial(n=r[:, k], p=p_m[:, k], size=(1, num_variables))

    return theta_C, r, p, seq_depth, data, states


def generate_easy_data(rng, m_samples=M, const_seq_depth=False):
    if const_seq_depth:
        seq_depth = np.ones(m_samples)
    else:
        seq_depth = rng.poisson(lam=2.5, size=m_samples)
        seq_depth[seq_depth == 0] = 1
    
    theta_C = np.array([0.35, 0.65])
    r1 = np.array([80, 50])
    r2 = np.array([50, 80])
    r = np.concatenate((r1[:, None], r2[:, None]), axis=1)
    p1 = np.array([0.3, 0.6])
    p2 = np.array([0.6, 0.3])
    p = np.concatenate((p1[:, None], p2[:, None]), axis=1)

    data = np.zeros((m_samples, 2))
    states = np.zeros(m_samples)
    for m in range(m_samples):
        p_seq_depth = p / (p + seq_depth[m] - seq_depth[m] * p)
        k = rng.choice(np.arange(2), p=theta_C)
        states[m] = k
        data[m] = rng.negative_binomial(n=r[:, k], p=p_seq_depth[:, k], size=(1, 2))
    
    return theta_C, r, p, seq_depth, data, states


# ---------------------------- EM initialization utils
def categorical_initialization(rng):
    theta_C_init = rng.binomial(30, 0.5, K)
    theta_C_init = theta_C_init / theta_C_init.sum()
    theta_X_init = rng.binomial(30, 0.5, (N, K, D))
    theta_X_init = theta_X_init / theta_X_init.sum(axis=2)[:, :, None]
    return theta_C_init, theta_X_init

def get_init_from_gmm(gmm_model):
    means = gmm_model.means_
    cov = gmm_model.covariances_
    r_mm = means ** 2 / (cov - means)
    p_mm = means / cov
    theta_c = gmm_model.weights_
    return theta_c, r_mm.T, p_mm.T


def gmm_init(rng, num_states, data, seq_depth, r_max):
    if seq_depth is None:
        seq_depth = np.ones(data.shape[0])
    train_data = data / seq_depth[:, None]
    gmm_normalized = GaussianMixture(n_components=num_states, covariance_type="diag", random_state=rng.binomial(100, 0.5)).fit(train_data)
    gmm = GaussianMixture(n_components=num_states, covariance_type="diag", random_state=rng.binomial(100, 0.5)).fit(data)
    theta_c_norm, r_norm, p_norm = get_init_from_gmm(gmm_normalized)
    _, r_unnormalized, p_unnormalized = get_init_from_gmm(gmm)
    gmm_unnorm_wrong_values = np.sum(r_unnormalized <= 0)
    if DEBUG:
        print(f"GMM unnormalized wrong values: {gmm_unnorm_wrong_values}")
    r_unnormalized[r_unnormalized <= 0] = rng.uniform(low=1, high=r_max, size=gmm_unnorm_wrong_values)
    p_unnormalized[np.logical_or(p_unnormalized <= 0, p_unnormalized >= 1)] = rng.uniform(0, 1, np.sum(np.logical_or(p_unnormalized <= 0, p_unnormalized >= 1)))
    gmm_norm_wrong_values = np.sum(r_norm <= 0)
    if DEBUG:
        print(f"GMM normalized wrong values: {gmm_norm_wrong_values}")
    r_norm[r_norm <= 0] = r_unnormalized[r_norm <= 0]
    p_norm[np.logical_or(p_norm <= 0, p_norm >= 1)] = p_unnormalized[np.logical_or(p_norm <= 0, p_norm >= 1)]
    return theta_c_norm, r_norm, p_norm


def random_init(rng, r_max, repeat_r_p, random_noise=True, num_variables=N, num_states=K):
    r_sig = r_max / 20
    p_sig = 0.0001

    # theta_C_init = rng.binomial(100, 0.5, num_states)
    theta_C_init = rng.uniform(size=num_states)
    theta_C_init = theta_C_init / theta_C_init.sum()

    if repeat_r_p:
        p_val = rng.uniform(0, 1, (num_variables, 1))
        r_val = rng.uniform(low=1, high=r_max, size=(num_variables, 1))
        r_init = np.repeat(r_val, num_states, axis=1)
        p_init = np.repeat(p_val, num_states, axis=1)
        if random_noise:
            r_init = r_init + rng.normal(0, r_sig, (num_variables, num_states))
            p_init = p_init + rng.normal(0, p_sig, (num_variables, num_states))

        # keep parameters constraints
        r_init[r_init <= 0] = r_max // 2
        p_init[p_init >= 1] = 0.9
        p_init[p_init <= 0] = 0.1
    else:
        p_init = rng.uniform(0.2, 0.8, (num_variables, num_states))
        r_init = rng.binomial(100, 0.5, (num_variables, num_states))

    return theta_C_init, r_init, p_init


def init_param_old_em(rng, old_theta_C, old_theta_X, num_variables):
    if old_theta_C is None:
        return None, None
    r, p = old_theta_X
    init_theta_C = np.append(old_theta_C, 1e-5)
    init_theta_C = init_theta_C / init_theta_C.sum()
    if len(old_theta_C) > 1:
        r2 = np.hstack((r, rng.uniform(low=1, high=(np.max(r)), size=(num_variables, 1))))
    else:
        r2 = np.hstack((r, (np.arange(num_variables) + 1)[:, None]))
    p2 = np.hstack((p, np.repeat(0.5, num_variables)[:, None]))
    init_theta_X = np.array([r2, p2])
    return init_theta_C, init_theta_X


def init_params(rng, data=None, seq_depth=None, random_noise=True,
                repeat_r_p=True, r_max=100, num_variables=N, num_states=K,
                old_theta_C=None, old_theta_X=None, init_type="gmm"):
    if init_type == "gmm":
        theta_C_init, r_init, p_init = gmm_init(rng, num_states,
                                                data, seq_depth, r_max)
    elif init_type == "random":
        theta_C_init, r_init, p_init = random_init(rng, r_max, random_noise, repeat_r_p,
                                                   num_variables, num_states)
    elif init_type == "old":
        theta_C_init, theta_X_init = init_param_old_em(rng, old_theta_C,
                                                       old_theta_X, num_variables)
        r_init, p_init = theta_X_init
    else:
        theta_C_init, r_init, p_init = None, None, None
    return theta_C_init, r_init, p_init


# ---------------------------- data loading and saving utils

def load_data(path):
    data = np.load("data/" + path + "/data.npy")
    states = np.load("data/" + path + "/states.npy")
    real_theta_C = np.load("data/" + path + "/real_theta_C.npy")
    real_r = np.load("data/" + path + "/real_r.npy")
    real_p = np.load("data/" + path + "/real_p.npy")
    theta_C_init, r_init, p_init, seq_depth = None, None, None, None
    if os.path.exists("data/" + path + "/theta_C_init.npy"):
        theta_C_init = np.load("data/" + path + "/theta_C_init.npy")
    if os.path.exists("data/" + path + "/r_init.npy"):
        r_init = np.load("data/" + path + "/r_init.npy")
    if os.path.exists("data/" + path + "/p_init.npy"):
        p_init = np.load("data/" + path + "/p_init.npy")
    if os.path.exists("data/" + path + "/seq_depth.npy"):
        seq_depth = np.load("data/" + path + "/seq_depth.npy")
    return real_theta_C, real_r, real_p, seq_depth, data, states, theta_C_init, r_init, p_init


def save_seed(seed: str, M: int, N: int, K: int):
    file_name = datetime.now(pytz.timezone('Israel')).strftime("%d-%m-%Y_%H-%M-%S")
    with open(f"data/{file_name}.txt", 'w') as f:
        f.write(seed)
        f.write(f"M = {M}")
        f.write(f"N = {N}")
        f.write(f"K = {K}")


def save_data(real_theta_C, real_r, real_p, data, states, seq_depth=None, theta_C_init=None, r_init=None, p_init=None, prefix=None):
    if not prefix:
        prefix = datetime.now(pytz.timezone('Israel')).strftime("%d-%m-%Y_%H-%M-%S")
    os.mkdir(f"data/{prefix}")
    np.save(f"data/{prefix}/data.npy", data)
    np.save(f"data/{prefix}/states.npy", states)
    np.save(f"data/{prefix}/real_theta_C.npy", real_theta_C)
    np.save(f"data/{prefix}/real_r.npy", real_r)
    np.save(f"data/{prefix}/real_p.npy", real_p)
    if seq_depth is not None:
        np.save(f"data/{prefix}/seq_depth.npy", seq_depth)
    if theta_C_init is not None:
        np.save(f"data/{prefix}/theta_C_init.npy", theta_C_init)
    if r_init is not None:
        np.save(f"data/{prefix}/r_init.npy", r_init)
    if p_init is not None:
        np.save(f"data/{prefix}/p_init.npy", p_init)


# ---------------------------- evaluation utils

def calc_S_count(data, states, num_states, _D):
    _, num_variables = data.shape
    S_count = np.zeros((num_variables, num_states, _D))
    for i in range(num_variables):
        for k in range(num_states):
            for d in range(_D):
                S_count[i, k, d] = data[(data[:, i] == d) & (states == k)].shape[0]
    return S_count


# ---------------------------- plots & visualization utils

def plot_phi(phi_data, root=None, max_range=50, samples=100, y_lim=None):
    right = (2 * root) if (root is not None and root > max_range) else max_range
    X = np.arange(0, right, step=(right / samples))
    phi_X = [phi_data(x) for x in X]
    plt.plot(X, phi_X)
    plt.title("no root")
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black") 
    if root:
        plt.title(f"$phi$(root={root})={phi_data(root)}")
        plt.axvline(x=root, c="red")
    plt.show()


def plot_easy_dataset(data, states, output_r=None, output_p=None, real_r=None, real_p=None):
    plt.title("Small datasets with two states, and two features")

    x_c0 = np.array(data[states == 0])
    # plt.scatter(x=x_c0[:, 0], y=x_c0[:, 1], c="blue", alpha=0.25, edgecolors="none", label="state 1")
    # plt.plot([0, np.mean(x_c0[:, 0])], [0, np.mean(x_c0[:, 1])], color="black")
    # plt.plot([np.mean(x_c0[:, 0])], [np.mean(x_c0[:, 1])], marker="o", markersize=5, color="black")

    x_c1 = np.array(data[states == 1])
    # plt.scatter(x=x_c1[:, 0], y=x_c1[:, 1], c="orange", alpha=0.25, edgecolors="none", label="state 2")
    # plt.plot([0, np.mean(x_c1[:, 0])], [0, np.mean(x_c1[:, 1])], color="black")
    # plt.plot([np.mean(x_c1[:, 0])], [np.mean(x_c1[:, 1])], marker="o", markersize=5, color="black", label="mean")

    if output_r is not None and output_p is not None:
        output_means = output_r * ((1-output_p) / output_p)
        plt.plot([0, output_means[0, 0]], [0, output_means[1, 0]], color="grey")
        plt.plot([np.mean(output_means[0, 0])], [output_means[1, 0]], marker="o", markersize=5, color="grey")
        plt.plot([0, output_means[0, 1]], [0, output_means[1, 1]], color="grey")
        plt.plot([np.mean(output_means[0, 1])], [output_means[1, 1]], marker="o", markersize=5, color="grey", label="output parameters mean")

    if real_r is not None and real_r is not None:
        real_param_means = real_r * ((1-real_p) / real_p)
        plt.plot([0, real_param_means[0, 0]], [0, real_param_means[1, 0]], color="brown")
        plt.plot([np.mean(real_param_means[0, 0])], [real_param_means[1, 0]], marker="o", markersize=5, color="brown")
        plt.plot([0, real_param_means[0, 1]], [0, real_param_means[1, 1]], color="brown")
        plt.plot([np.mean(real_param_means[0, 1])], [real_param_means[1, 1]], marker="o", markersize=5, color="brown", label="real parameters mean")

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.show()


def plot_grad(grad, xs=None, i=None, k=None):
    xs = np.array(xs) if xs is not None else np.array([])
    r_range = np.arange(1, 100)
    p_range = np.arange(1, 100) / 100
    grad_result = np.zeros((r_range.shape[0], p_range.shape[0]))
    for i_r in range(r_range.shape[0]):
        for i_p in range(p_range.shape[0]):
            grad_result[i_r][i_p] = np.linalg.norm(grad((r_range[i_r], p_range[i_p])))
    m = np.array(np.meshgrid(p_range, r_range)).reshape((2, -1)).T
    plt.figure(figsize=(10, 10))
    # plt.scatter(x=m[:,0], y=m[:,1], c=grad_result.clip(0, 1e5), vmin=0, cmap="YlOrRd")
    plt.scatter(x=m[:, 0], y=m[:, 1], c=np.log(grad_result), cmap="YlOrRd")
    plt.colorbar()
    if xs is not None and i is not None and k is not None:
        plt.scatter(xs[:, 1], xs[:, 0], s=36, color="blue")
        grads_xs = [grad(x) for x in xs]
        min_norm = np.min(np.linalg.norm(grads_xs, axis=1))
        plt.title(f"logarithm of norm of the gradient of a single CPD {i}, {k} (blue), and convergence of different initializations (orange)\n min grad_result = {min_norm}")
    else:
        min_grad = np.min(grad_result)
        plt.title(f"logarithm of norm of the gradient of a single CPD, min grad = {min_grad}")
    plt.xlabel("$p$")
    plt.ylabel("$r$")
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    plt.show()


def variance_plot(data):
    var_vec = np.var(data, axis=0)
    plt.hist(np.log(var_vec))
    plt.show()


def bic_score(ll, k, n):
    return -2 * ll + k * np.log(n)


def free_parameters_NB(k, N):
    cpd_parameters = 2 * k * N
    prior_parameters = k - 1
    return prior_parameters + cpd_parameters


def savefig_and_data(fig, plot_folder, figname,  *args):
    work_folder = f"{plot_folder}/{figname}"
    if not os.path.exists(work_folder):
        os.mkdir(work_folder)
    fig.savefig(fname=f"{work_folder}/{figname}")
    np.savez(f"{work_folder}/data_file", *args)


def calc_accuracy_for_perm(log_posterior_mat, states, perm):
    log_posterior_mat_perm = log_posterior_mat[perm, :]
    predicted_states = np.argmax(log_posterior_mat_perm, axis=0)
    accuracy = np.mean(np.array(states) == np.array(predicted_states))
    return accuracy


# ---------------------------- testing utils

def test_grad_p(func, derive, p_range, r_val, func_name):
    epsilon = 1e-7
    derive_numerical_func = lambda p: (func(r_val, p + epsilon) - func(r_val, p)) / epsilon

    derivative_numerical_vals = np.array([derive_numerical_func(p_val) for p_val in p_range])
    derivative_analytical_vals = np.array([derive(r_val, p_val) for p_val in p_range])

    B = 1e9
    plt.scatter(derivative_numerical_vals / B, derivative_analytical_vals / B)
    plt.title(f"derivation of {func_name} with respect to p")
    plt.xlabel("numerical derivative")
    plt.ylabel("analytical derivative")
    plt.show()


def test_grad_r(func, derive, r_range, p_val, func_name):
    epsilon = 1e-7
    derive_numerical_func = lambda r: (func(r + epsilon, p_val) - func(r, p_val)) / epsilon

    derivative_numerical_vals = np.array([derive_numerical_func(r_val) for r_val in r_range])
    derivative_analytical_vals = np.array([derive(r_val, p_val) for r_val in r_range])

    plt.plot(derivative_numerical_vals, derivative_analytical_vals)
    plt.title(f"derivation of {func_name} with respect to r")
    plt.xlabel("numerical derivative")
    plt.ylabel("analytical derivative")
    plt.show()


# ---------------------------- data preprocessing utils

def load_real_data(data_path, seq_depth_path, tag_path):
    data = pd.read_csv(data_path)
    seq_depth = np.loadtxt(seq_depth_path, delimiter=",")
    tags = np.loadtxt(tag_path, delimiter=",", dtype=str)
    return [data, seq_depth, tags]


def test_train_split(data_folder):
    # load data from r
    data_path = data_folder + "data.csv"
    seq_depth_path = data_folder + "seq_depth.csv"
    tag_path = data_folder + "tags.csv"
    data, seq_depth, tags = load_real_data(data_path, seq_depth_path, tag_path)

    NK_index = np.where(tags == "NK")[0]
    not_NK_index = np.where(tags != "NK")[0]
    test_index_NK = np.random.choice(NK_index, replace=False)
    train_index_NK = np.delete(NK_index, np.where(NK_index == test_index_NK)[0][0])

    test_index_not_NK = np.random.choice(not_NK_index, int(np.floor(not_NK_index.shape[0] * 0.2)), replace=False)
    train_index_not_NK = np.delete(not_NK_index, test_index_not_NK)

    test_index = np.append(test_index_not_NK, test_index_NK)
    train_index = np.append(train_index_not_NK, train_index_NK)

    test_data = data.iloc[test_index.tolist(), ]
    train_data = data.iloc[train_index.tolist(), ]

    test_seq_depth = seq_depth[test_index]
    train_seq_depth = seq_depth[train_index]

    test_tags = tags[test_index]
    train_tags = tags[train_index]

    train_data.to_csv(data_folder + "train.csv", index=False)
    test_data.to_csv(data_folder + "test.csv", index=False)
    np.savetxt(data_folder + "train_seq_depth.csv", train_seq_depth, delimiter=",")
    np.savetxt(data_folder + "test_seq_depth.csv", test_seq_depth, delimiter=",")
    np.savetxt(data_folder + "train_tags.csv", train_tags, delimiter=",", fmt="%s")
    np.savetxt(data_folder + "test_tags.csv", test_tags, delimiter=",", fmt="%s")


def _absolute_gap_from_second(means_sort, means_i, variances_i):
    criterion = np.abs(means_i[means_sort[0]] - means_i[means_sort[1]]) > ABSOLUTE_GAP_FROM_SECOND
    return criterion


def _absolute_gap_from_third(means_sort, means_i, variances_i):
    criterion = np.abs(means_i[means_sort[1]] - means_i[means_sort[2]]) < ABSOLUTE_GAP_FROM_THIRD
    return criterion


def _log_fold_change(means_sort, means_i, variances_i):
    criterion = np.abs(np.log2(1 + means_i[means_sort[0]]) - np.log2(1 + means_i[means_sort[1]])) > FOLD_CHANGE_THRESHOLD
    return criterion


def _cv_threshold(means_sort, means_i, variances_i):  # coefficient of variation
    cv = variances_i[means_sort[0]] / means_i[means_sort[0]]
    criterion = 0 < cv < VAR_THRESHOLD
    return criterion


def significance_criterion(means_i, variances_i):
    means_sort = np.argsort(-means_i)
    if (OVER_EXPRESSED_ONLY is False and
            np.abs(means_i[means_sort[0]] - means_i[means_sort[1]]) <
            np.abs(means_i[means_sort[-2]] - means_i[means_sort[-1]])):
        means_sort = means_sort[::-1]

    criteria = [
                _absolute_gap_from_second,
                _absolute_gap_from_third,
                _log_fold_change
                # _cv_threshold,  # only a few genes passes the last criteria, so don't need to add another filter
                ]
    is_significant = np.all([criterion(means_sort, means_i, variances_i) for criterion in criteria])
    if is_significant:  # and np.all(means_i > 5):
        cluster = means_sort[0]
        # rank = np.abs(means_i[means_sort[0]] - means_i[means_sort[1]]) / np.sqrt(variances_i[means_sort[0]])
        rank = np.abs(np.log2(1 + means_i[means_sort[0]]) - np.log2(1 + means_i[means_sort[1]]))
        return True, cluster, rank
    else:
        return False, None, None


def find_significant_genes(data, num_clusters, rng):
    M, N = data.shape

    # calc means and variances for each gene and each cluster
    kmeans = KMeans(n_clusters=num_clusters, random_state=rng.binomial(100, 0.5)).fit(data)
    kmeans_prediction = kmeans.predict(data)
    means = kmeans.cluster_centers_
    variances = np.array([np.var(data[kmeans_prediction == k], axis=0) for k in range(num_clusters)])

    # calc the significance for each gene
    significant_genes_per_cluster = [[] for _ in range(num_clusters)]
    for i in tqdm(range(N)):
        is_significant, cluster, rank = significance_criterion(means[:, i], variances[:, i])
        if is_significant:
            significant_genes_per_cluster[cluster].append([i, rank])

    for c in range(num_clusters):
        significant_genes_per_cluster[c].sort(key=lambda gene: -gene[1])

    return significant_genes_per_cluster, kmeans_prediction


def load_processed_real_data(execute_feature_selection=True,
                             ):
    # -------------------- load data
    train_data = pd.read_csv(DATA_LOCATION + "train.csv")
    train_seq_depth = np.loadtxt(DATA_LOCATION + "train_seq_depth.csv", delimiter=",")
    test_data = pd.read_csv(DATA_LOCATION + "test.csv")
    test_seq_depth = np.loadtxt(DATA_LOCATION + "test_seq_depth.csv", delimiter=",")
    train_tags = []
    with open(DATA_LOCATION + "train_tags.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                train_tags.append(row[0])
    test_tags = []
    with open(DATA_LOCATION + "test_tags.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                test_tags.append(row[0])

    # -------------------- filter smaller than 1 sequencing depth
    # right now we cannot handle seq_depth < 1
    train_data = train_data[train_seq_depth >= 1]
    train_tags = [train_tags[i] for i in range(len(train_tags)) if (train_seq_depth >= 1)[i]]
    train_seq_depth = train_seq_depth[train_seq_depth >= 1]
    test_data = test_data[test_seq_depth >= 1]
    test_tags = [test_tags[i] for i in range(len(test_tags)) if (test_seq_depth >= 1)[i]]
    test_seq_depth = test_seq_depth[test_seq_depth >= 1]

    # -------------------- feature selection
    selected_genes = []
    if execute_feature_selection:
        filename = f"selected_genes_over_exp_only_{OVER_EXPRESSED_ONLY}.csv"
        with open(DATA_LOCATION + filename) as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    selected_genes.append(row[0])
        train_data = train_data[selected_genes]
        test_data = test_data[selected_genes]

    return train_data, train_seq_depth, train_tags, test_data, test_seq_depth, \
           test_tags, selected_genes
