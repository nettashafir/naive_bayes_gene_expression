from utils import *

from em_naive_bayes_categorical import (e_step_Categorical,
                                        m_step_categorical,
                                        calc_logpmf_mat_categorical,
                                        compute_mle_categorical,
                                        evaluate_simulated_data_categorical)
from em_naive_bayes_negative_binomial import (e_step_negative_binomial,
                                              m_step_negative_binomial,
                                              calc_logpmf_mat_NegativeBinomial,
                                              compute_empirical_mle_negative_binomial,
                                              compute_empirical_mm_negative_binomial,
                                              evaluate_simulated_data_negative_binomial,
                                              evaluate_real_data_negative_binomial)
from em_naive_bayes_negative_binomial_seq_depth import (e_step_negative_binomial_sequencing_depth,
                                                        m_step_negative_binomial_sequencing_depth,
                                                        calc_logpmf_mat_negative_binomial_sequencing_depth,
                                                        compute_empirical_mle_negative_binomial_seq_depth,
                                                        compute_empirical_mm_negative_binomial_seq_depth)

EM_COMPONENTS = {CATEGORICAL_EM_CODE: ("Categorical",
                                       e_step_Categorical,
                                       m_step_categorical,
                                       calc_logpmf_mat_categorical),
                 NEGATIVE_BINOMIAL_EM_CODE: ("Negative Binomial",
                                             e_step_negative_binomial,
                                             m_step_negative_binomial,
                                             calc_logpmf_mat_NegativeBinomial),
                 NEGATIVE_BINOMIAL_SEQ_DEPTH_EM_CODE: ("Negative Binomial Sequencing Depth",
                                                       e_step_negative_binomial_sequencing_depth,
                                                       m_step_negative_binomial_sequencing_depth,
                                                       calc_logpmf_mat_negative_binomial_sequencing_depth)}

EVALUATION_COMPONENTS = {CATEGORICAL_EM_CODE: (compute_mle_categorical,
                                               compute_mle_categorical),
                         NEGATIVE_BINOMIAL_EM_CODE: (compute_empirical_mle_negative_binomial,
                                                     compute_empirical_mm_negative_binomial),
                         NEGATIVE_BINOMIAL_SEQ_DEPTH_EM_CODE: (compute_empirical_mle_negative_binomial_seq_depth,
                                                               compute_empirical_mm_negative_binomial_seq_depth)}


# ------------------------------------ EM variants

def NaiveBayesEM(data: np.ndarray,
                 theta_C_init: np.ndarray,
                 theta_X_init: np.ndarray,
                 seq_depth: np.ndarray             = None,
                 rng: np.random.Generator          = None,
                 max_iter: int                     = DEF_MAX_ITER,
                 print_logs: bool                  = True,
                 epsilon: Optional[float]          = STOP_ITERATION_EPSILON,
                 cpd_code: int                     = NEGATIVE_BINOMIAL_SEQ_DEPTH_EM_CODE,
                 ):
    assert seq_depth is None or data.shape[0] == seq_depth.shape[0], "data and the sequencing depth are not in the same shape"
    assert not (seq_depth is None and cpd_code == NEGATIVE_BINOMIAL_SEQ_DEPTH_EM_CODE), "sequencing depth is missing"
    rng = np.random.default_rng(DEF_SEED) if rng is None else rng
    CPD_distribution, E_step, M_step, calc_logpmf_mat = EM_COMPONENTS[cpd_code]
    if cpd_code == CATEGORICAL_EM_CODE:
        d = D
        d_description = "support size"
    else:
        d = int(np.max(data))
        d_description = "max(data)"

    M, N = data.shape
    K = theta_C_init.shape[0]
    if print_logs:
        print("\n========== start EM ==========\n")
        print(f"CPD's distribution: {CPD_distribution}")
        print(f"Dimensions:"
              f"\n\tM (n_samples) = {M}"
              f"\n\tN (n_variables) = {N}"
              f"\n\tK (n_states) = {K}"
              f"\n\tD ({d_description}) = {d}",
              end="\n\n")

    likelihoods = []
    log_marginal, log_posterior = calc_log_marginal_and_posterior(calc_logpmf_mat, data, theta_C_init, theta_X_init, seq_depth=seq_depth)
    normalized_log_marginal_init = np.sum(log_marginal) / (N * M)
    likelihoods.append(normalized_log_marginal_init)
    theta_C, theta_X = theta_C_init, theta_X_init
    if print_logs:
        print(f"init: marginal log-likelihood = {likelihoods[-1]}")

    for i in range(max_iter):
        ESS_C, ESS_X = E_step(data, log_posterior)
        theta_C, theta_X = M_step(ESS_C, ESS_X, M, seq_depth=seq_depth, theta_X_last=theta_X, rng=rng)

        log_marginal, log_posterior = calc_log_marginal_and_posterior(calc_logpmf_mat, data, theta_C, theta_X, seq_depth=seq_depth)
        normalized_log_marginal = np.sum(log_marginal) / (N * M)
        likelihoods.append(normalized_log_marginal)

        if print_logs:
            print(f"iteration {i+1}: marginal log-likelihood = {likelihoods[-1]}")

        there_is_a_bug = likelihoods[-1] < likelihoods[-2] and not np.isclose(likelihoods[-1], likelihoods[-2])
        assert not there_is_a_bug, "The likelihood cannot get worse"

        if epsilon and (likelihoods[-1] - likelihoods[-2] < epsilon):
            break

    if print_logs:
        print("\n========== end EM ==========\n\n")
    return theta_C, theta_X, likelihoods


def em_all_params(data, theta_C_init, theta_X_init, seq_depth, rng,
                  calc_logpmf_mat=calc_logpmf_mat_negative_binomial_sequencing_depth,
                  E_step=e_step_negative_binomial_sequencing_depth,
                  M_step=m_step_negative_binomial_sequencing_depth,
                  n_iter=DEF_MAX_ITER):
    M, N = data.shape
    K = theta_C_init.shape[0]
    likelihoods = []
    log_marginal, log_posterior = calc_log_marginal_and_posterior(calc_logpmf_mat, data, theta_C_init, theta_X_init, seq_depth=seq_depth)
    normalized_log_marginal_init = np.sum(log_marginal) / (N * M)
    likelihoods.append(normalized_log_marginal_init)
    theta_C, theta_X = theta_C_init, theta_X_init
    theta_C_array = np.empty((n_iter + 1, K))
    r_array = np.empty((n_iter + 1, N, K))
    p_array = np.empty((n_iter + 1, N, K))
    theta_C_array[0] = theta_C_init
    r_array[0] = theta_X_init[0]
    p_array[0] = theta_X_init[1]
    for i in range(1, n_iter + 1):
        print(f"iteration {i}: log-likelihood = {likelihoods[-1]}")
        ESS_C, ESS_X = E_step(data, log_posterior)
        theta_C, theta_X = M_step(ESS_C, ESS_X, M, seq_depth, rng, theta_X_last=theta_X)
        theta_C_array[i] = theta_C
        r_array[i] = theta_X[0]
        p_array[i] = theta_X[1]
        log_marginal, log_posterior = calc_log_marginal_and_posterior(calc_logpmf_mat, data, theta_C, theta_X, seq_depth=seq_depth)
        normalized_log_marginal = np.sum(log_marginal) / (N * M)
        likelihoods.append(normalized_log_marginal)
        assert not (likelihoods[-1] < likelihoods[-2] and not np.isclose(likelihoods[-1], likelihoods
            [-2])), "The likelihood cannot get worse"
    return theta_C_array, r_array, p_array, likelihoods


def run_EM_n_times(n_times, rng, data, real_theta_C, real_theta_X, states, seq_depth, cpd_code, r_max=100, init_type="gmm"):
    real_r, real_p = real_theta_X
    normalized_theoretical_log_marginal_likelihood = calc_normalized_log_marginal_likelihood(
        EM_COMPONENTS[cpd_code][3], data, real_theta_C, real_theta_X,
        seq_depth=seq_depth)
    print(f"theoretical log-likelihood = {normalized_theoretical_log_marginal_likelihood}")
    c_list = []
    r_list = []
    p_list = []
    for i in range(n_times):
        theta_C_init, r_init, p_init = init_params(rng, data, seq_depth,
                                                   random_noise=RANDOM_NOISE_INIT,
                                                   r_max=r_max, init_type=init_type)
        theta_X_init = np.array([r_init, p_init])
        c1, x1, ll = NaiveBayesEM(data, theta_C_init, theta_X_init, seq_depth,
                                  max_iter=15, rng=rng, cpd_code=cpd_code)
        r1, p1 = x1
        evaluate_simulated_data_negative_binomial(data, rng,
                                                  real_theta_C, theta_C_init, c1,
                                                  real_r, r_init, r1,
                                                  real_p, p_init, p1,
                                                  states=states,
                                                  compute_empirical_MLE=EVALUATION_COMPONENTS[cpd_code][0],
                                                  compute_empirical_MM=EVALUATION_COMPONENTS[cpd_code][1],
                                                  seq_depth=seq_depth)
    return c_list, r_list, p_list


# ------------------------------------ Main

def main_real_data(seed, cpd_code=CPD_CODE):
    # (!) execute the file "preprocessing.py" before executing this function
    rng = np.random.default_rng(seed)

    # -------------------- load data
    train_data = pd.read_csv(DATA_LOCATION + "train.csv")
    train_seq_depth = np.loadtxt(DATA_LOCATION + "train_seq_depth.csv", delimiter=",")
    train_tags = []
    with open(DATA_LOCATION + "train_tags.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                train_tags.append(row[0])
    b, states = np.unique(train_tags, return_inverse=True)

    # right now we cannot handle seq_depth < 1
    train_data = train_data[train_seq_depth >= 1]
    train_tags = np.array(train_tags)[train_seq_depth >= 1]
    train_seq_depth = train_seq_depth[train_seq_depth >= 1]

    # -------------------- feature selection
    selected_genes = []
    with open(DATA_LOCATION + "selected_genes.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                selected_genes.append(row[0])
    train_data_filtered = train_data[selected_genes]

    # -------------------- run EM
    num_samples, num_variables = train_data_filtered.shape
    num_states = np.unique(states).shape[0]
    theta_C_init, r_init, p_init = init_params(rng=rng,
                                               random_noise=RANDOM_NOISE_INIT,
                                               num_variables=num_variables,
                                               num_states=num_states)
    theta_X_init = np.array([r_init, p_init])
    output_theta_C, output_theta_X, likelihoods = NaiveBayesEM(train_data_filtered.values,
                                                               theta_C_init, theta_X_init, train_seq_depth,
                                                               rng=rng, max_iter=15, print_logs=True,
                                                               cpd_code=cpd_code)
    output_r, output_p = output_theta_X

    # -------------------- Evaluate
    evaluate_real_data_negative_binomial(train_data_filtered.values,
                                         theta_C_init, output_theta_C,
                                         r_init, output_r,
                                         p_init, output_p,
                                         states=states,
                                         compute_empirical_MLE=compute_empirical_mle_negative_binomial_seq_depth,
                                         seq_depth=train_seq_depth,
                                         rng=rng)

    # -------------------- save seed & model parameters
    if SAVE_RESULTS:
        save_seed(seed=str(seed), num_samples=M, num_variables=N, num_states=K)


def main_simulated_data(seed, cpd_code=CPD_CODE, r_max=100):
    rng = np.random.default_rng(seed)
    rng_generate_data = np.random.default_rng(DEF_SEED_GENERATE_DATA)
    seq_depth = None

    # -------------------- load or generate data
    if cpd_code == CATEGORICAL_EM_CODE:
        real_theta_C, real_theta_X, data, states = generate_categorical_data(rng)
    else:
        seq_depth = np.ones(M) if cpd_code < 2 else None
        # (!) Negative Binomial converge to Normal distribution when r grows (!)
        real_theta_C, real_r, real_p, seq_depth, data, states = \
            generate_data_NegativeBinomial(rng=rng_generate_data,
                                           seq_depth=seq_depth,
                                           r_max=r_max)

    # -------------------- run EM
    if cpd_code == CATEGORICAL_EM_CODE:
        theta_C_init, theta_X_init = categorical_initialization(rng)
    else:
        theta_C_init, r_init, p_init = init_params(rng=rng,
                                                   random_noise=RANDOM_NOISE_INIT,
                                                   num_variables=N,
                                                   num_states=K)
        theta_X_init = np.array([r_init, p_init])
    output_theta_C, output_theta_X, likelihoods = NaiveBayesEM(data, theta_C_init, theta_X_init, seq_depth,
                                                               max_iter=100, rng=rng, cpd_code=cpd_code)

    # -------------------- Evaluate
    if cpd_code == CATEGORICAL_EM_CODE:
        evaluate_simulated_data_categorical(data, states,
                                            real_theta_C, theta_C_init, output_theta_C,
                                            real_theta_X, theta_X_init, output_theta_X,
                                            compute_empirical_MLE=compute_mle_categorical)
    else:
        output_r, output_p = output_theta_X
        evaluate_simulated_data_negative_binomial(data, rng,
                                                  real_theta_C, theta_C_init, output_theta_C,
                                                  real_r, r_init, output_r,
                                                  real_p, p_init, output_p,
                                                  states=states,
                                                  compute_empirical_MLE=EVALUATION_COMPONENTS[cpd_code][0],
                                                  compute_empirical_MM=EVALUATION_COMPONENTS[cpd_code][1],
                                                  seq_depth=seq_depth)

    # -------------------- save data & init parameters
    if SAVE_RESULTS:
        save_seed(seed=str(seed), num_samples=M, num_variables=N, num_states=K)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATA_LOCATION = sys.argv[1]
    start = time.time()
    if REAL_DATA:
        print(" ---      run on real data     ---")
        main_real_data(DEF_SEED)
    else:
        print(" ---   run on simulated data   ---")
        main_simulated_data(DEF_SEED)
    end = time.time()
    seconds = end - start
    print(f" --- total time for EM {seconds // 60} minutes and {seconds % 60} seconds --- ")
