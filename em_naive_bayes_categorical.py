from utils import *

def calc_logpmf_mat_categorical(data, theta_C, theta_X, *args, **kwargs):
    """
        Returns a matrix logpmf_mat in shape (K, M), where logpmf_mat[k, m] = log(p(x[m] | C=k))
    """
    M, N = data.shape
    _, K, D = theta_X.shape

    matrix_index = np.repeat(np.arange(N)[None, :], M, axis=0).T
    matrix_index_tensor = np.repeat(matrix_index[None, :], K, axis=1).reshape((N, K, M))
    rows_index = np.repeat(np.arange(K)[None, :], M, axis=0).T
    rows_index_tensor = np.repeat(rows_index[None, :], N, axis=0)
    col_index_tensor = np.repeat(data.T, K, axis=0).reshape((N, K, M)).astype(int)
    pmf_tensor = theta_X[matrix_index_tensor, rows_index_tensor, col_index_tensor]
    # logpmf_mat = np.log(reduce(np.multiply, pmf_tensor))
    logpmf_mat = reduce(np.add, np.log(pmf_tensor))
    return logpmf_mat


# ------------------------------------ E-step

def _calc_sufficient_X(data, posteriors_matrix):
    M, N = data.shape
    K, _ = posteriors_matrix.shape
    assert D > np.max(data), f"D = {D} is not large enough"

    matrix_index = np.repeat(np.arange(N)[None, :], M, axis=0).T
    matrix_index_tensor = np.repeat(matrix_index[:, None], K, axis=1)
    rows_index = np.repeat(np.arange(K)[None, :], M, axis=0).T
    rows_index_tensor = np.repeat(rows_index[None, :], N, axis=0)
    col_index_tensor = np.repeat(data.T[:, None], K, axis=1).astype(int)
    posterior_tensor = np.repeat(posteriors_matrix[None, :], N, axis=0)

    S_X = np.zeros((N, K, D))
    np.add.at(S_X, [matrix_index_tensor, rows_index_tensor, col_index_tensor], posterior_tensor)
    return S_X


def e_step_Categorical(data, log_posterior):
    # --- Calculate sufficient statistic for C, which is a vector with length K.
    M, N = data.shape
    K, _ = log_posterior.shape
    ESS_C = np.exp(LSE(log_posterior, axis=1))

    # --- Calculate sufficient statistic for X, which is a tensor with shape (N, K, D).
    posterior_matrix = np.exp(log_posterior)
    ESS_X = _calc_sufficient_X(data, posterior_matrix)

    return ESS_C, ESS_X


# ------------------------------------ M-step

def m_step_categorical(ESS_C, ESS_X, M, *args, **kwargs):
    # --- Calculate theta_X, which is a vector with length K.
    theta_C = ESS_C / M
    assert np.isclose(np.sum(theta_C), 1), f"Theta_C is not a legal distribution, for it's sum is {np.sum(theta_C)}"

    # --- Calculate theta_X, whose shape is (N, K, D).
    theta_X = ESS_X / ESS_C[:, None]

    return theta_C, theta_X


# ------------------------------------ Evaluation functions

def compute_mle_categorical(data, states, num_states):
    """
        Calculate empirical MLE with real Sufficient Statistics.
        This is also the estimation by the Method if Moments
    """
    num_samples, num_variables = data.shape

    S_C = np.histogram(states, bins=num_states, range=(0, num_states))[0]
    S_X = calc_S_count(data, states, num_states, D)
    theta_C_MLE, theta_X_MLE = m_step_categorical(S_C, S_X.astype(int), num_samples)
    return theta_C_MLE, theta_X_MLE

def evaluate_simulated_data_categorical(data, states,
                                        real_theta_C, theta_C_init, output_theta_C,
                                        real_theta_X, theta_X_init, output_theta_X,
                                        compute_empirical_MLE=None):
    num_states = real_theta_C.shape[0]

    perm = np.arange(num_states)
    empirical_theta_C, empirical_theta_X = None, None
    if states is not None:
        if compute_empirical_MLE is not None:
            empirical_theta_C, empirical_theta_X = compute_empirical_MLE(data, states, num_states)
            orig, perm = linear_sum_assignment(- np.outer(empirical_theta_C, output_theta_C))  # find the order

    print("========== Evaluation ==========\n")
    print(f"true theta_C:\n {real_theta_C}")
    if compute_empirical_MLE is not None:
        print(f"empirical theta_C (MLE using real sufficient statistics):\n {empirical_theta_C}")
    # print(f"init theta_C:\n {theta_C_init}")
    print(f"output theta_C:\n {output_theta_C[perm]}", end="\n\n")

    print("true theta_X:\n", real_theta_X[0, :HEAD_EVALUATE, :HEAD_EVALUATE])
    if compute_empirical_MLE is not None:
        print(f"empirical theta_C (MLE using real sufficient statistics):\n {empirical_theta_X[0, :HEAD_EVALUATE, :HEAD_EVALUATE]}")
    # print("r init:\n", r_init)
    print("output theta_X:\n", output_theta_X[0, :HEAD_EVALUATE, :HEAD_EVALUATE], end="\n\n")

