from utils import *


# ------------------------------------ Negative Binomial functions

def log_likelihood_ik(S_C_k, S_sum_ik, S_count_ik):
    """
    Return a function of the conditional likelihood of a single CPD, given expected sufficient statistics (ESS).
    The returned function is the parts of the complete log likelihood that depends on r_ik and p_ik.
    """
    def inner(r_ik, p_ik):
        return np.sum(S_count_ik * loggamma(np.arange(S_count_ik.shape[0]) + r_ik)) \
               + S_C_k * (r_ik * np.log(p_ik) - loggamma(r_ik)) \
               + S_sum_ik * np.log(1 - p_ik)
    return inner


def grad_r_ik(S_C_k: float, S_sum_ik: float, S_count_ik: np.ndarray):
    """
    Returns a function which is the first derivative of the complete log-likelihood with respect to r_ik
    """
    def inner(r_ik):
        return S_C_k * (np.log(r_ik) + np.log(S_C_k) - np.log(S_sum_ik + r_ik * S_C_k) - digamma(r_ik)) + np.sum(
            S_count_ik * digamma(np.arange(S_count_ik.shape[0]) + r_ik))

    return inner


def hessian_r_ik(S_C_k, S_sum_ik, S_count_ik):
    """
    Returns a function which is the second derivative of the complete log-likelihood with respect to r_ik
    """
    def inner(r_ik):
        return S_C_k * (1 / r_ik - (S_C_k / (S_sum_ik + S_C_k * r_ik)) - polygamma(1, r_ik)) + np.sum(
            S_count_ik * polygamma(1, (np.arange(S_count_ik.shape[0]) + r_ik)))

    return inner


def calc_logpmf_mat_NegativeBinomial(data, theta_C, theta_X, *args, **kwargs):
    """
    Returns a matrix logpmf_mat in shape (K, M), where logpmf_mat[k, m] = log(p(x[m] | C=k))
    This function is equivalent to:

        logpmf_mat = np.zeros((K, M))
        for k in range(K):
            logpmf_mat[k] = np.sum(scipy.stats.nbinom.logpmf(data, r[:, k], p[:, k], loc=0), axis=1)

    The *args and **kwargs are for respecting the API of the EM that includes seq_depth argument
    """
    M, N = data.shape
    K = theta_C.shape[0]

    r, p = theta_X
    r_rep = np.repeat(r.flatten(), M, axis=0).reshape((N, K, M))  # r_rep[i, k, :] = r[i, k]
    p_rep = np.repeat(p.flatten(), M, axis=0).reshape((N, K, M))  # p_rep[i, k, :] = p[i, k]
    x_rep = np.repeat(data.T, K, axis=0).reshape((N, K, M))       # x_rep[i, :, m] = data[m, i]

    first_term = loggamma(x_rep + r_rep)    # (N, K, M), first_term[i, k, m] = loggamma(data[m, i] + r[i, k])
    second_term = loggamma(x_rep + 1)       # (N, K, M), second_term[i, k, m] = loggamma(data[m, i] + 1)
    third_term = loggamma(r_rep)            # (N, K, M), third_term[i, k, m] = loggamma(r[i, k])
    forth_term = r_rep * np.log(p_rep)      # (N, K, M), forth_term[i, k, m] = r[i, k] * log(p[i, k])
    fifth_term = x_rep * np.log(1 - p_rep)  # (N, K, M), fifth_term[i, k, m] = data[m, i] * log(1 - p[i, k])
    all_together = first_term - second_term - third_term + forth_term + fifth_term

    logpmf_mat = np.sum(all_together, axis=0)  # (K, M), log_likelihood_mat[k, m] = log(p(x[m] | C=k))

    if np.any(np.isnan(logpmf_mat)):
        raise ValueError("logpmf matrix has nan entries - there is p which is 0 or 1")

    return logpmf_mat


# ------------------------------------ E-step

def e_step_negative_binomial(data, log_posterior):
    M, N = data.shape
    K, M = log_posterior.shape

    # Calculate expected sufficient statistics
    log_sufficient_C = LSE(log_posterior, axis=1)
    ESS_C = np.exp(log_sufficient_C)

    x_rep = np.repeat(data.T, K, axis=0).reshape((N, K, M))
    log_posterior_rep = np.repeat(log_posterior[None, :], N, axis=0).reshape((N, K, M))
    log_ESS_sum = LSE(np.log(x_rep) + log_posterior_rep, axis=2)
    ESS_sum = np.exp(log_ESS_sum)

    log_ESS_count = calc_log_ESS_count(data, log_posterior)
    ESS_count = np.exp(log_ESS_count)

    ESS_X = [ESS_sum, ESS_count]
    return ESS_C, ESS_X


# ------------------------------------ M-step
def _r_solver_root(log_likelihood, grad):
    msg, res = "", None
    full_res: RootResults = brentq(f=grad, a=1e-10, b=1e3, full_output=True)[1]
    res = full_res.root
    if res == 0:
        raise ValueError("illegal value of r == 0")
    msg = "found root"
    return msg, res


def _r_solver_maximizer(log_likelihood, grad):
    minus_ll = lambda x: -log_likelihood(*x)
    full_res: OptimizeResult = minimize(minus_ll, x0=np.array([1, 0.5]), jac=grad, bounds=[(0, np.inf), (0, 1)], method='Nelder-Mead')
    res = full_res.x[0]
    msg = "got to local maximum" if full_res.success else full_res.message
    return msg, res


def _r_solver_root_and_maximizer(log_likelihood, grad):
    minus_ll = lambda x: -log_likelihood(*x)
    try:
        full_res: RootResults = brentq(f=grad, a=1e-10, b=1e3, full_output=True)[1]
        res = full_res.root
        if res == 0:
            raise ValueError("illegal value of r == 0")
        msg = "found root"
    except ValueError as e:
        msg = f"ERROR: {str(e)}"
        full_res: OptimizeResult = minimize(minus_ll, x0=np.array([1, 0.5]), jac=grad, method='Nelder-Mead')
        res = full_res.x[0]
        additional_msg = "got to local maximum" if full_res.success else full_res.message
        msg += f" -> {additional_msg}"
    return msg, res


def m_step_negative_binomial(ESS_C, ESS_X, M, theta_X_last=None, *args, **kwargs):
    ESS_sum, ESS_count = ESS_X
    N, K = ESS_sum.shape
    # N, K, D = ESS_count.shape

    theta_C = ESS_C / M

    if THETA_C_SMOOTHING_EPSILON is not None and np.any(ESS_C < 1):
        if DEBUG:
            print("SMOOTHING THETA_C!")
        theta_C += THETA_C_SMOOTHING_EPSILON  # in the plots it's 1e-5
        theta_C /= theta_C.sum()
    assert np.isclose(np.sum(theta_C), 1), f"Theta_C is not a legal distribution, for it's sum is {np.sum(theta_C)}"
    # assert not (np.any(np.isclose(theta_C, 0)) or np.any(np.isclose(theta_C, 1))), f"Some state get extreme probability: min(theta_C)={np.min(theta_C)}, max(theta_C)={np.max(theta_C)}"

    r, p = theta_X_last if theta_X_last is not None else [np.zeros((N, K)), np.zeros((N, K))]

    for k in range(K):
        for i in range(N):
            msg = ""

            log_likelihood_func = log_likelihood_ik(S_C_k=ESS_C[k], S_sum_ik=ESS_sum[i, k], S_count_ik=ESS_count[i, k])
            log_likelihood_last = log_likelihood_func(r[i, k], p[i, k]) if theta_X_last is not None else -np.inf
            grad = grad_r_ik(ESS_C[k], ESS_sum[i, k], ESS_count[i, k, :])
            # hessian = hessian_r_ik(ESS_C[k], ESS_sum[i, k], ESS_count[i, k, :])

            try:
                # find r
                msg, new_r_ik = _r_solver_root_and_maximizer(log_likelihood_func, grad)

                # find p
                numerator = new_r_ik * ESS_C[k]
                denominator = ESS_sum[i, k] + new_r_ik * ESS_C[k]
                if np.isclose(denominator, 0) or np.isclose(numerator, 0) or np.isclose(ESS_sum[i, k], 0):
                    raise ValueError("p gets illegal value of 0 or 1")
                new_p_ik = numerator / denominator

                # check if the complete likelihood raised (the root finder can converge but make the likelihood worse)
                got_improvement = log_likelihood_func(new_r_ik, new_p_ik) > log_likelihood_last
                if got_improvement:
                    r[i, k] = new_r_ik
                    p[i, k] = new_p_ik
                    msg += " -> got improvement in likelihood"


            except ValueError as e:
                msg += f" -> ERROR: {str(e)}"

            finally:
                if DEBUG:
                    print(f"CPD ({i}, {k}): {msg}")

    theta_X = np.array([r, p])
    return theta_C, theta_X


# ------------------------------------ Evaluation functions

def compute_empirical_mle_negative_binomial(data, states, num_states, *args):
    """
        Calculate empirical MLE with real Sufficient Statistics
    """
    num_samples, num_variables = data.shape

    S_C = np.histogram(states, bins=num_states, range=(0, num_states))[0]
    S_sum = np.zeros((num_variables, num_states))
    for k in range(num_states):
        S_sum[:, k] = np.sum(data[states == k], axis=0)
    _D = int(np.max(data))
    S_count = calc_S_count(data, states, num_states, _D)
    empirical_theta_C_MLE, empirical_theta_X_MLE = m_step_negative_binomial(S_C, [S_sum, S_count.astype(int)], num_smaples)
    empirical_r_MLE, empirical_p_MLE = empirical_theta_X_MLE
    return empirical_theta_C_MLE, empirical_r_MLE, empirical_p_MLE


def compute_empirical_mm_negative_binomial(data, states, K, **kwargs):
    """
        Calculate moment estimation for r and p
    """
    M, N = data.shape

    empirical_theta_C_MM = np.histogram(states, bins=K, range=(0, K))[0] / M

    empirical_r_MM, empirical_p_MM = np.zeros((N, K)), np.zeros((N, K))
    for k in range(K):
        data_k = data[states == k]
        means = np.mean(data_k, axis=0)
        vars = np.var(data_k, axis=0)
        empirical_p_MM[:, k] = means / vars
        empirical_r_MM[:, k] = (means ** 2) / (vars - means)

    return empirical_theta_C_MM, empirical_r_MM, empirical_p_MM


def evaluate_simulated_data_negative_binomial(data, rng,
                                              real_theta_C, theta_C_init, output_theta_C,
                                              real_r, r_init, output_r,
                                              real_p, p_init, output_p,
                                              states=None, compute_empirical_MLE=None, compute_empirical_MM=None,
                                              **kwargs
                                              ):
    K = real_theta_C.shape[0]

    perm = np.arange(K)
    empirical_theta_C_MLE, empirical_r_MLE, empirical_p_MLE = None, None, None
    if states is not None:
        if compute_empirical_MLE is not None:
            empirical_theta_C_MLE, empirical_r_MLE, empirical_p_MLE = compute_empirical_MLE(data, states, K, rng=rng,
                                                                                            **kwargs)
            # orig, perm = linear_sum_assignment(- empirical_r_MLE.T @ output_r)  # find the order
            orig, perm = linear_sum_assignment(- np.outer(empirical_theta_C_MLE, output_theta_C))  # find the order

    print("========== Evaluation ==========\n")
    print(f"true theta_C:\n {real_theta_C}")
    if compute_empirical_MLE is not None:
        print(f"empirical theta_C (MLE using real sufficient statistics):\n {empirical_theta_C_MLE}")
    # print(f"init theta_C:\n {theta_C_init}")
    print(f"output theta_C:\n {output_theta_C[perm]}", end="\n\n")

    print("true r:\n", real_r[:HEAD_EVALUATE, :])
    if compute_empirical_MLE is not None:
        print(f"empirical r (MLE using real sufficient statistics):\n {empirical_r_MLE[:HEAD_EVALUATE, :]}")
    # print("r init:\n", r_init)
    print("output r:\n", output_r[:HEAD_EVALUATE, perm], end="\n\n")

    print("true p:\n", real_p[:HEAD_EVALUATE, :])
    if compute_empirical_MLE is not None:
        print(f"empirical p (MLE using real sufficient statistics):\n {empirical_p_MLE[:HEAD_EVALUATE, :]}")
    # print("p init:\n", p_init)
    print("output p:\n", output_p[:HEAD_EVALUATE, perm], end="\n\n")


def evaluate_real_data_negative_binomial(data,
                                         theta_C_init, output_theta_C,
                                         r_init, output_r,
                                         p_init, output_p,
                                         states=None, compute_empirical_MLE=None,
                                         **kwargs):
    K = theta_C_init.shape[0]

    perm = np.arange(K)
    if states is not None:
        if compute_empirical_MLE is not None:
            empirical_theta_C_MLE, empirical_r_MLE, empirical_p_MLE = compute_empirical_MLE(data, states, K, **kwargs)
            # orig, perm = linear_sum_assignment(
            #     - empirical_r_MLE.T @ output_r)  # find the order

    print("========== Evaluation ==========\n")
    if compute_empirical_MLE is not None:
        print(f"empirical theta_C (MLE using real sufficient statistics):\n {empirical_theta_C_MLE}")
    print(f"init theta_C:\n {theta_C_init}")
    print(f"output theta_C:\n {output_theta_C[perm]}", end="\n\n")

    if compute_empirical_MLE is not None:
        print(f"empirical r (MLE using real sufficient statistics):\n {empirical_r_MLE[:HEAD_EVALUATE, :]}")
    print("r init:\n", r_init[:HEAD_EVALUATE, :])
    print("output r:\n", output_r[:HEAD_EVALUATE, perm], end="\n\n")

    if compute_empirical_MLE is not None:
        print(f"empirical p (MLE using real sufficient statistics):\n {empirical_p_MLE[:HEAD_EVALUATE, :]}")
    print("p init:\n", p_init[:HEAD_EVALUATE, :])
    print("output p:\n", output_p[:HEAD_EVALUATE, perm], end="\n\n")
