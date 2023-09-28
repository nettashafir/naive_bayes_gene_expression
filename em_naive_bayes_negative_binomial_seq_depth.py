from utils import *


# ------------------------------------ Negative Binomial functions
def calc_logpmf_mat_negative_binomial_sequencing_depth(data, theta_C, theta_X, seq_depth):
    """
    Returns a matrix logpmf_mat in shape (K, M), where logpmf_mat[k, m] = log(p(x[m] | C=k))
    """
    M, N = data.shape
    K = theta_C.shape[0]
    r, p = theta_X

    # TODO find more efficient implementation
    logpmf_mat = np.zeros((K, M))
    for m in range(M):
        p_seq_dep = p / (p + seq_depth[m] - seq_depth[m] * p)
        for k in range(K):
            logpmf_mat[k, m] = np.sum(nbinom.logpmf(data[m], r[:, k], p_seq_dep[:, k], loc=0))

    return logpmf_mat


def log_likelihood_ik(log_posterior_k, S_C_k, S_count_ik, data_i, seq_depth):
    """
    Return a function of the conditional likelihood of a single CPD, given expected sufficient statistics (ESS).
    The returned function is the parts of the complete log likelihood that depends on r_ik and p_ik.
    """
    def inner(r_ik, p_ik):
        return np.sum(S_count_ik * loggamma(np.arange(S_count_ik.shape[0]) + r_ik)) \
               - S_C_k * loggamma(r_ik) \
               - r_ik * np.exp(LSE(log_posterior_k + np.log(np.log(p_ik + (1 - p_ik) * seq_depth) - np.log(p_ik)))) \
               + np.log(1 - p_ik) * np.exp(LSE(log_posterior_k + np.log(data_i))) \
               - np.exp(LSE(log_posterior_k + np.log(data_i) + np.log(np.log(p_ik + (1 - p_ik) * seq_depth))))
    return inner


def f2_ik(indicators_k, seq_depth, data_i):
    """
    The parts of likelihood function that depends on p_ik
    """
    def inner(r_ik, p_ik):
        return r_ik * np.sum(indicators_k * (np.log(p_ik) - np.log(p_ik + (1 - p_ik) * seq_depth))) + \
               np.sum(indicators_k * data_i * (np.log(1-p_ik) - np.log(p_ik + (1 - p_ik) * seq_depth)))
    return inner


def f3_ik(log_posterior_k, seq_depth, S_C_k, S_count_ik):
    """
    The parts of likelihood function that depends on r_ik
    """
    def inner(r_ik, p_ik):
        return np.sum(S_count_ik * loggamma(np.arange(S_count_ik.shape[0]) + r_ik)) \
                - S_C_k * loggamma(r_ik) \
                - np.exp(LSE(log_posterior_k + np.log(r_ik) +
                             np.log(np.log(p_ik + (1 - p_ik) * seq_depth) -
                                    np.log(p_ik))))
    return inner


# ------------------------------------ E-step

def e_step_negative_binomial_sequencing_depth(data, log_posterior):
    """
    TODO: document
    """
    log_sufficient_C = LSE(log_posterior, axis=1)
    ESS_C = np.exp(log_sufficient_C)

    log_ESS_count = calc_log_ESS_count(data, log_posterior)
    ESS_count = np.exp(log_ESS_count)

    ESS_X = [ESS_count, log_posterior, data]
    return ESS_C, ESS_X


# ------------------------------------ M-step

def grad_p_ik(log_posterior_k, seq_depth, data_i):
    def inner(r_ik, p_ik):
        return (r_ik / p_ik) * np.exp(
            LSE(log_posterior_k + np.log(seq_depth) - np.log(
                p_ik + (1 - p_ik) * seq_depth))) \
               - (1 / (1 - p_ik)) * np.exp(
            LSE(log_posterior_k + np.log(data_i) - np.log(
                p_ik + (1 - p_ik) * seq_depth)))
    return inner


def grad_r_ik(log_posterior_k, seq_depth, S_C_k, S_count_ik):
    def inner(r_ik, p_ik):
        return np.sum(
            S_count_ik * digamma(np.arange(S_count_ik.shape[0]) + r_ik)) \
               - S_C_k * digamma(r_ik) \
               - np.exp(LSE(log_posterior_k +
                            np.log(np.log(p_ik + (1 - p_ik) * seq_depth) -
                                   np.log(p_ik))))
    return inner


def grad_ik(log_posterior_k, seq_depth, data_i, S_C_k, S_count_ik):
    grad_r_func = grad_r_ik(log_posterior_k, seq_depth, S_C_k, S_count_ik)
    grad_p_func = grad_p_ik(log_posterior_k, seq_depth, data_i)
    def inner(x_ik):
        r_ik, p_ik = x_ik
        return np.array([grad_r_func(r_ik, p_ik), grad_p_func(r_ik, p_ik)])
    return inner


def hessian_r_ik(S_C_k, S_count_ik):
    def inner(r_ik, p_ik):
        return np.sum(S_count_ik * polygamma(1, np.arange(S_count_ik.shape[0]) + r_ik)) \
               - S_C_k * polygamma(1, r_ik)
    return inner


def hessian_rp_ik(log_posterior_k, seq_depth):
    def inner(r_ik, p_ik):
        return (1 / p_ik) * np.exp(LSE(log_posterior_k + np.log(seq_depth) - np.log(p_ik + (1 - p_ik) * seq_depth)))
    return inner


def hessian_p_ik(log_posterior_k, seq_depth, data_i, S_C_k):
    def inner(r_ik, p_ik):
        seq_depth_helper = ((1 - seq_depth) / (p_ik + (1 - p_ik) * seq_depth)) ** 2
        data_helper = log_posterior_k + np.log(data_i)
        return (-r_ik / (p_ik ** 2)) * S_C_k \
               + r_ik * np.exp(LSE(log_posterior_k + np.log(seq_depth_helper)))\
               - (1 / ((1 - p_ik) ** 2)) * np.exp(LSE(data_helper)) \
               + np.exp(LSE(data_helper + np.log(seq_depth_helper)))
    return inner

def hessian_ik(log_posterior_k, seq_depth, data_i, S_C_k, S_count_ik):
    H_00 = hessian_r_ik(S_C_k, S_count_ik)
    H_10 = hessian_rp_ik(log_posterior_k, seq_depth)
    H_11 = hessian_p_ik(log_posterior_k, seq_depth, data_i, S_C_k)
    def inner(x_ik):
        r_ik, p_ik = x_ik
        return (np.array([[H_00(r_ik, p_ik), H_10(r_ik, p_ik)],
                         [H_10(r_ik, p_ik), H_11(r_ik, p_ik)]])
                + 1e-10 * np.eye(2))
    return inner


def _solver_root(grad, inits, hessian, log_likelihood, log_likelihood_last):
    success, msg, root_res = False, "", None  # for annoying interpreters
    for j in range(len(inits)):
        root_res = root(fun=grad, x0=inits[j], jac=hessian, method="krylov")
        legal_solution = root_res.x[0] > 0 and 0 < root_res.x[1] < 1
        got_improvement = log_likelihood(*root_res.x) > log_likelihood_last
        if legal_solution and got_improvement:
            if root_res.success:
                msg = "found root"
            else:
                msg = "got improvement in likelihood by another point than the root"
            success = True
            break
    return success, msg, root_res


def _solver_maximize(grad, inits, hessian, log_likelihood_func, log_likelihood_last):
    minus_ll = lambda x: -log_likelihood_func(*x)
    success, msg, res = False, "", None  # for annoying interpreters
    for j in range(len(inits)):
        res = minimize(minus_ll, inits[j], jac=grad, hess=hessian, bounds=[(0, np.inf), (0, 1)], method='Nelder-Mead')
        got_improvement = log_likelihood_func(*res.x) > log_likelihood_last
        if got_improvement:
            if res.success:
                msg = "got to local maximum"
            else:
                msg = "got improvement in likelihood by another point than the root"
            success = True
            break
    return success, msg, res


def _solver_root_and_maximize(grad, inits, hessian, log_likelihood_func, log_likelihood_last):
    minus_ll = lambda x: -log_likelihood_func(*x)
    success, msg, res = False, "", None  # for annoying interpreters
    res = root(fun=grad, x0=inits[0], jac=hessian, method="krylov")
    legal_solution = res.x[0] > 0 and 0 < res.x[1] < 1
    got_improvement = log_likelihood_func(*res.x) > log_likelihood_last
    if legal_solution and got_improvement:
        if res.success:
            msg = "found root"
        else:
            msg = "got improvement in likelihood by another point than the root (root finder)"
        success = True
    else:
        res = minimize(minus_ll, inits[0], jac=grad, hess=hessian, bounds=[(0, np.inf), (0, 1)], method='Nelder-Mead')
        got_improvement = log_likelihood_func(*res.x) > log_likelihood_last
        if got_improvement:
            if res.success:
                msg = "got to local maximum"
            else:
                msg = "got improvement in likelihood by another point than the root (maximizer)"
            success = True
    return success, msg, res


def m_step_negative_binomial_sequencing_depth(ESS_C, ESS_X, M, seq_depth, rng, theta_X_last=None):
    K = ESS_C.shape[0]
    ESS_count, log_posterior, data = ESS_X
    M, N = data.shape

    # ----- compute parameters
    theta_C = ESS_C / M
    r, p = theta_X_last if theta_X_last is not None else [np.zeros((N, K)), np.zeros((N, K))]

    root_res, success, msg = None, False, ""  # for annoying interpreters
    for i in range(N):
        for k in range(K):
            try:
                log_likelihood_func = log_likelihood_ik(log_posterior_k=log_posterior[k], S_C_k=ESS_C[k], S_count_ik=ESS_count[i, k], data_i=data[:, i], seq_depth=seq_depth)
                grad = grad_ik(log_posterior[k], seq_depth, data[:, i], ESS_C[k], ESS_count[i, k])
                hessian = hessian_ik(log_posterior[k], seq_depth, data[:, i], ESS_C[k], ESS_count[i, k])
                log_likelihood_last = log_likelihood_func(r[i, k], p[i, k]) if theta_X_last is not None else -np.inf
                inits = [[rng.uniform(1, 100), rng.uniform(0, 1)] for _ in range(N_ROOT_FINDER_INITIALIZATIONS)]
                success, msg, root_res = _solver_maximize(grad, inits, hessian, log_likelihood_func, log_likelihood_last)
                if success:
                    r[i, k], p[i, k] = root_res.x
                else:
                    msg = f"the solver didn't converged after {N_ROOT_FINDER_INITIALIZATIONS} initializations"

            except ValueError as e:
                msg = f"ERROR: {str(e)}"

            finally:
                if DEBUG:
                    print(f"CPD ({i}, {k}): {msg}")

    theta_X = np.array([r, p])
    return theta_C, theta_X


# ------------------------------------ Evaluation functions

def compute_empirical_mle_negative_binomial_seq_depth(data, states, num_states, seq_depth, rng):
    """
        Calculate empirical MLE for r and p with real sufficient statistics and sequencing depth
    """
    num_samples, num_variables = data.shape

    S_C = np.histogram(states, bins=num_states, range=(0, num_states))[0]
    S_count = calc_S_count(data, states, num_states, int(np.max(data)))
    log_posterior = np.full((num_states, num_samples), -np.inf)
    for m in range(num_samples):
        log_posterior[int(states[m]), m] = 0
    empirical_theta_C_MLE, empirical_theta_X_MLE = m_step_negative_binomial_sequencing_depth(S_C, [S_count, log_posterior, data], num_samples, seq_depth, rng=rng)
    empirical_r_MLE, empirical_p_MLE = empirical_theta_X_MLE
    return empirical_theta_C_MLE, empirical_r_MLE, empirical_p_MLE


def compute_empirical_mm_negative_binomial_seq_depth(data, states, K, seq_depth):
    """
        Calculate empirical moment estimation for r and p with real sufficient statistics and sequencing depth
    """
    empirical_theta_C_MM = np.histogram(states, bins=K, range=(0, K))[0] / M

    empirical_r_MM, empirical_p_MM = np.zeros((N, K)), np.zeros((N, K))
    for k in range(K):
        for i in range(N):
            data_ki = data[states == k, i]
            seq_depth_k = seq_depth[states == k]
            mean_estimator = np.mean(data_ki / seq_depth_k)

            n_bins = len(np.unique(np.array(seq_depth, dtype=int)))
            bin_vec = np.linspace(np.min(seq_depth), np.max(seq_depth), num=n_bins)
            which_bin = np.zeros(len(seq_depth_k))
            for b in range(n_bins-1):
                which_bin += b*np.logical_and(seq_depth_k > bin_vec[b], seq_depth_k < bin_vec[b+1])
            eta_var_estimator = np.array([np.var(data_ki[which_bin == b], ddof=1) for b in range(n_bins)])
            var_estimator = np.mean((eta_var_estimator[which_bin.astype(int)] + (seq_depth_k - 1)*data_ki) / seq_depth_k**2)

            empirical_r_MM[i, k] = (mean_estimator ** 2) / (var_estimator - mean_estimator)
            empirical_p_MM[i, k] = mean_estimator / var_estimator

    return empirical_theta_C_MM, empirical_r_MM, empirical_p_MM
