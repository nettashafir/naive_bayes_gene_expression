from main import *


# -------------------------------------------- algorithm checks plots

def _compute_loglikelihood_mle(data, states, seq_depth, rng,
                               cpd_code=NEGATIVE_BINOMIAL_SEQ_DEPTH_EM_CODE):
    compute_mle_function = EVALUATION_COMPONENTS[cpd_code][0]
    log_pmf_function = EM_COMPONENTS[cpd_code][3]
    empirical_theta_C, empirical_r, empirical_p = compute_mle_function(data, states, K, seq_depth, rng)
    empirical_log_marginal, _ = calc_log_marginal_and_posterior(log_pmf_function, data, empirical_theta_C,
        [empirical_r, empirical_p], seq_depth=seq_depth)
    return empirical_log_marginal


def create_parameter_plot(parameter_name, parameter_df, parameter_df_mle, n_runs, N, K):
    title_fontsize, xlabel_fontsize = 20, 15
    r_p_figsize = (15, 10)
    fig, axs = plt.subplots(N, K, figsize=r_p_figsize)
    fig.suptitle(
        f"${parameter_name}$ for N={N} and K={K} as a Function of Iteration, for "
        f"{n_runs} Different Runs", fontsize=title_fontsize)
    fig.text(0.5, 0.04, 'Iteration', ha='center',
             fontsize=xlabel_fontsize)
    fig.text(0.04, 0.5, parameter_name, va='center', fontsize=title_fontsize,
             rotation="vertical")
    for i in range(N):
        for k in range(K):
            axs[i, k].axhline(y=parameter_df[i, k], c="blue",
                              label=f"theoretical ${parameter_name}$")
            axs[i, k].axhline(y=parameter_df_mle[i, k], c="forestgreen",
                              label=f"empirical ${parameter_name}$ (MLE)")
    return fig, axs


def plot_p_per_iteration(rng, data, states, include_init, random_noise, repeat,
                         true_p, seq_depth, cpd_code, n_runs, n_iters):
    M, N = data.shape
    compute_mle_function = EVALUATION_COMPONENTS[cpd_code][0]
    log_pmf_function = EM_COMPONENTS[cpd_code][3]
    empirical_theta_C_MLE, empirical_r_MLE, empirical_p_MLE = compute_mle_function(
        data, states, K, seq_depth, rng)
    fig_p, axs_p = create_parameter_plot("p", true_p, empirical_p_MLE, n_runs,
                                         N, K)

    x_axis = np.arange(n_iters + 1)
    all_p = np.zeros((n_runs, n_iters + 1, N, K))

    for run_id in tqdm(range(n_runs)):
        error = True
        while error:
            try:
                theta_C_init, r_init, p_init = init_params(rng, data, seq_depth,
                                                           random_noise=random_noise,
                                                           repeat_r_p=repeat,
                                                           init_type="random")
                _, outputs_r, outputs_p, _ = em_all_params(data, theta_C_init,
                                                           [r_init, p_init],
                                                           seq_depth=seq_depth,
                                                           rng=rng,
                                                           n_iter=n_iters)
            except Exception as e:
                print(str(e))
            else:
                error = False

        if not include_init:
            outputs_r = outputs_r[1:]
            outputs_p = outputs_p[1:]
            x_axis = np.arange(1, n_iters + 1)

        orig, perm = linear_sum_assignment(- empirical_r_MLE.T @ outputs_r[-1])
        all_p[run_id] = outputs_p

        for k in range(K):
            for i in range(N):
                axs_p[i, k].plot(x_axis, outputs_p[:, :, perm][:, i, k],
                                 c="black")

    final_p = all_p[:, -1, :, :]
    rmse_p_matrix = np.sqrt(np.mean((final_p - empirical_p_MLE) ** 2, axis=0))
    if include_init:
        # labels = [*range(n_iters + 1)]
        # labels[1] = 'init'
        for k in range(K):
            for i in range(N):
                # axs_p[i, k].set_xticklabels(labels)
                axs_p[i, k].set_title(f"RMSE: {np.round(rmse_p_matrix[i, k], 2)}")
    fig_p.subplots_adjust(hspace=0.4)
    plt.show()


def plot_parameters_and_likelihood_per_iteration(rng, data, seq_depth, states, real_theta_C,
                                                 real_r, real_p,
                                                 n_iters=50, n_runs=1, include_init=True,
                                                 cpd_code=NEGATIVE_BINOMIAL_SEQ_DEPTH_EM_CODE):
    K = real_theta_C.shape[0]
    M, N = data.shape
    theta_C_figsize = (9, 6)
    likelihood_figsize = (12, 6)
    title_fontsize = 20
    xlabel_fontsize = 15
    compute_mle_function = EVALUATION_COMPONENTS[cpd_code][0]
    log_pmf_function = EM_COMPONENTS[cpd_code][3]
    empirical_theta_C_MLE, empirical_r_MLE, empirical_p_MLE = compute_mle_function(data, states, K, seq_depth, rng)
    # compute_mm_function = EVALUATION_COMPONENTS[cpd_code][1]
    # empirical_theta_C_MM, empirical_r_MM, empirical_p_MM = compute_mm_function(data, states, K, seq_depth)

    # Creates likelihood plot
    fig_likelihoods, axs_likelihoods = plt.subplots(figsize=likelihood_figsize)
    theoretical_log_marginal, _ = calc_log_marginal_and_posterior(log_pmf_function, data, real_theta_C, [real_r, real_p], seq_depth=seq_depth)
    normalized_theoretical_log_marginal = np.sum(theoretical_log_marginal) / (N * M)
    axs_likelihoods.axhline(y=normalized_theoretical_log_marginal, c="blue",
                            label="theoretical normalized log marginal")
    empirical_MLE_log_marginal, _ = calc_log_marginal_and_posterior(log_pmf_function, data, empirical_theta_C_MLE, [empirical_r_MLE, empirical_p_MLE], seq_depth=seq_depth)
    normalized_empirical_MLE_log_marginal = np.sum(empirical_MLE_log_marginal) / (N * M)
    axs_likelihoods.axhline(y=normalized_empirical_MLE_log_marginal,
                            c="forestgreen",
                            label="empirical normalized log marginal (MLE)")
    # empirical_MM_log_marginal, _ = calc_log_marginal_and_posterior(
    #     calc_logpmf_mat_NegativeBinomialSequencingDepth, data, empirical_theta_C_MM, [empirical_r_MM, empirical_p_MM], seq_depth=seq_depth)
    # normalized_empirical_MM_log_marginal = np.sum(empirical_MM_log_marginal) / (N * M)
    # axs_likelihoods.axhline(y=normalized_empirical_MM_log_marginal,
    #                         c="limegreen",
    #                         label="empirical normalized log marginal (moments)")
    fig_likelihoods.suptitle(f"Changes in Likelihood as a Function of Iteration", fontsize=title_fontsize)
    axs_likelihoods.set_ylabel("Normalized Log Likelihood")
    axs_likelihoods.set_xlabel("Iteration")
    likelihood_at_first_iteration = []

    # Creates theta_C figure
    fig_theta_C, axs_theta_C = plt.subplots(K, 1, sharex=True,
                                            figsize=theta_C_figsize)
    axs_theta_C[0].set_title(f"$\u03B8^C$ as a Function of Iteration, "
                             f"for {n_runs} Different Runs",
                             fontsize=title_fontsize)
    for k in range(K):
        axs_theta_C[k].set_ylabel(f"$\u03B8^C_{k}$")
        axs_theta_C[k].axhline(y=real_theta_C[k], c="blue",
                               label=f"theoretical $\u03B8^C$")
        axs_theta_C[k].axhline(y=empirical_theta_C_MLE[k], c="green",
                               label=f"empirical $\u03B8^C$")
    axs_theta_C[-1].set_xlabel('Iteration', fontsize=xlabel_fontsize)

    # Creates r figure
    fig_r, axs_r = create_parameter_plot("r", real_r, empirical_r_MLE, n_runs, N, K)
    # Creates p figure
    fig_p, axs_p = create_parameter_plot("p", real_p, empirical_p_MLE, n_runs, N, K)

    # run EM
    x_axis = np.arange(n_iters + 1)
    all_theta_c = np.zeros((n_runs, n_iters + 1, K))
    all_r = np.zeros((n_runs, n_iters + 1, N, K))
    all_p = np.zeros((n_runs, n_iters + 1, N, K))
    for iter in tqdm(range(n_runs)):
        if iter == 0:
            theta_C_init, r_init, p_init = init_params(rng, data, seq_depth)
            outputs_theta_C, outputs_r, outputs_p, log_likelihoods = em_all_params(data, theta_C_init, [r_init, p_init], seq_depth=seq_depth, rng=rng, n_iter=n_iters)
            log_likelihoods = log_likelihoods[1:]
            axs_likelihoods.plot(x_axis[1:], log_likelihoods, c="red", label="gmm")
        else:
            theta_C_init, r_init, p_init = init_params(rng, data, seq_depth, init_type="random")
            outputs_theta_C, outputs_r, outputs_p, log_likelihoods = em_all_params(
                data, theta_C_init, [r_init, p_init], seq_depth=seq_depth,
                rng=rng, n_iter=n_iters)
            log_likelihoods = log_likelihoods[1:]
            if iter == 1:
                axs_likelihoods.plot(x_axis[1:], log_likelihoods, c="black", label="identical noised")
            else:
                axs_likelihoods.plot(x_axis[1:], log_likelihoods, c="black")

        if not include_init:
            outputs_theta_C = outputs_theta_C[1:]
            outputs_r = outputs_r[1:]
            outputs_p = outputs_p[1:]
            x_axis = np.arange(1, n_iters + 1)

        orig, perm = linear_sum_assignment(- empirical_r_MLE.T @ outputs_r[-1])
        likelihood_at_first_iteration.append(log_likelihoods[0])
        all_theta_c[iter] = outputs_theta_C
        all_r[iter] = outputs_r
        all_p[iter] = outputs_p

        for k in range(K):
            if iter == 0:
                axs_theta_C[k].plot(x_axis, outputs_theta_C[:, perm][:, k], c="red", label="gmm")
                for i in range(N):
                    axs_r[i, k].plot(x_axis, outputs_r[:, :, perm][:, i, k], c="red", label="gmm")
                    axs_p[i, k].plot(x_axis, outputs_p[:, :, perm][:, i, k], c="red", label="gmm")
            else:
                if iter == 1:
                    axs_theta_C[k].plot(x_axis, outputs_theta_C[:, perm][:, k], c="black",
                                         label="identical noised")
                    for i in range(N):
                        axs_r[i, k].plot(x_axis, outputs_r[:, :, perm][:, i, k], c="black", label="identical noised")
                        axs_p[i, k].plot(x_axis, outputs_p[:, :, perm][:, i, k], c="black", label="identical noised")
                else:
                    axs_theta_C[k].plot(x_axis, outputs_theta_C[:, perm][:, k],
                                        c="black")
                    for i in range(N):
                        axs_r[i, k].plot(x_axis, outputs_r[:, :, perm][:, i, k],
                                         c="black")
                        axs_p[i, k].plot(x_axis, outputs_p[:, :, perm][:, i, k],
                                         c="black")

    # edit x_ticks
    final_r = all_r[:, -1, :, :]
    final_p = all_p[:, -1, :, :]
    rmse_r_matrix = np.sqrt(np.mean((final_r - empirical_r_MLE) ** 2, axis=0))
    rmse_p_matrix = np.sqrt(np.mean((final_p - empirical_p_MLE) ** 2, axis=0))
    if include_init:
        labels = [item.get_text() for item in axs_theta_C[-1].get_xticklabels()]
        labels[1] = 'init'
        for k in range(K):
            axs_theta_C[k].set_xticklabels(labels)
            for i in range(N):
                axs_r[i, k].set_xticklabels(labels)
                axs_p[i, k].set_xticklabels(labels)
                axs_r[i, k].set_title(f"RMSE: {np.round(rmse_r_matrix[i, k], 2)}")
                axs_p[i, k].set_title(f"RMSE: {np.round(rmse_p_matrix[i, k], 2)}")

    # add legends
    handles_likelihood, labels_likelihood = axs_likelihoods.get_legend_handles_labels()
    axs_likelihoods.legend(handles_likelihood, labels_likelihood)
    handles_theta_C, labels_theta_C = axs_theta_C[0].get_legend_handles_labels()
    fig_theta_C.legend(handles_theta_C, labels_theta_C, loc='lower left')
    handles_r, labels_r = axs_r[0, 0].get_legend_handles_labels()
    fig_r.legend(handles_r, labels_r, loc='lower left')
    handles_p, labels_p = axs_p[0, 0].get_legend_handles_labels()
    fig_p.legend(handles_p, labels_p, loc='lower left')

    # y limits
    margin = 0.5
    upper_limit = np.max([normalized_theoretical_log_marginal, normalized_empirical_MLE_log_marginal]) + margin
    lower_limit = np.min(likelihood_at_first_iteration) - margin
    axs_likelihoods.set_ylim([lower_limit, upper_limit])

    # save figs
    suffix = "_uniform_noise_init" if RANDOM_NOISE_INIT else ""
    fig_r.subplots_adjust(hspace=0.4)
    fig_p.subplots_adjust(hspace=0.4)
    savefig_and_data(fig_likelihoods, "plots/",
                     f"likelihood1_per_iteration_with_{n_iters}_iters_{n_runs}_runs", log_likelihoods)
    savefig_and_data(fig_theta_C, "plots/",
                     f"theta_C1_per_iteration_with_{n_iters}_iters_{n_runs}_runs{suffix}", all_theta_c)
    savefig_and_data(fig_r, "plots/",
                     f"r1_per_iteration_with_{n_iters}_iters_{n_runs}_runs{suffix}", all_r)
    savefig_and_data(fig_p, "plots/",
                     f"p1_per_iteration_with_{n_iters}_iters_{n_runs}_runs{suffix}", all_p)
    plt.show()


def plot_likelihood_as_func_of_M(rng, M_list, n_iter=50, n_runs=5):
    # choose theta_c, r, p

    l = M_list.shape[0]
    ll_per_m = np.zeros((l, n_runs))
    for i in tqdm(range(l)):
        M = M_list[i]
        for j in range(n_runs):
            theta_C, r, p, seq_depth, data, states = generate_data_NegativeBinomial(rng, num_samples=M, num_variables=4, num_states=3)
            error = True
            counter = 0
            while error:
                if counter >= 10:
                    theta_C, r, p, seq_depth, data, states = generate_data_NegativeBinomial(rng, num_samples=M, num_variables=4, num_states=3)
                    counter = 0
                try:
                    theta_C_init, r_init, p_init = init_params(rng)
                    theta_X_init = [r_init, p_init]
                    log_likelihoods = NaiveBayesEM(data, theta_C_init, np.array(theta_X_init), seq_depth, max_iter=15, rng=rng, print_logs=False)[2]
                except Exception as e:
                    print(str(e))
                    counter += 1
                else:
                    error = False

            mle_log_likelihood_list = _compute_loglikelihood_mle(data, states, seq_depth, rng)
            mle_log_likelihood = np.sum(mle_log_likelihood_list) / (N * M)
            ll_per_m[i, j] = abs(mle_log_likelihood - log_likelihoods[-1])

    plt.figure(figsize=(12, 6))
    ll_flatten = ll_per_m.flatten()
    m_rep = M_list.repeat(n_runs)
    m_df = pd.DataFrame({"ll": ll_flatten, "M": m_rep})
    sns.boxplot(data=m_df, x="M", y="ll", color="grey")
    plt.legend()
    plt.title(f"Absoulte Difference Normalized LL as a Function of M with {n_runs} "
              f"Runs and Different Datasets")
    plt.ylabel("Absoulte difference between normalized LL and mle normalized LL")
    plt.xlabel("M")
    # plt.savefig(fname=f"likelihood_per_M_{n_runs}_runs")
    plt.show()


def plot_likelihood_as_func_of_N(rng, N_arr, seq_depth, n_iter=50, n_runs=5):
    l = N_arr.shape[0]
    ll_per_N = np.zeros((l, n_runs))
    for i in tqdm(range(l)):
        N = N_arr[i]
        data, states, real_theta_C, real_r, real_p = generate_data_NegativeBinomial(rng)
        for j in range(n_runs):
            error = True
            while error:
                try:
                    theta_C_init = np.random.binomial(30, 0.5, K)
                    theta_C_init = theta_C_init / theta_C_init.sum()
                    p_val = np.random.uniform(0, 1)
                    r_val = np.random.binomial(100, 0.5)
                    r_init = np.full((N, K), r_val)
                    p_init = np.full((N, K), p_val)
                    theta_X_init = np.array([r_init, p_init])
                    log_likelihoods = NaiveBayesEM(data, theta_C_init, theta_X_init, seq_depth, max_iter=15, rng=rng)[2]
                except:
                    pass
                else:
                    error = False
            ll_per_N[i, j] = log_likelihoods[-1]

    plt.figure(figsize=(12, 6))
    ll_flatten = ll_per_N.flatten()
    n_rep = N_arr.repeat(n_runs)
    n_df = pd.DataFrame({"ll": ll_flatten, "N": n_rep})
    sns.lineplot(data=n_df, x="N", y="ll")
    plt.legend()
    plt.title(f"normalized log likelihood as a function of N with {n_runs} "
              f"runs and same data")
    plt.ylabel("normalized log likelihood")
    plt.xlabel("N")
    # plt.savefig(fname=f"likelihood_per_M_{n_runs}_runs")
    plt.show()


def plot_easy_dataset(data, states):
    plt.title("Small datasets with two states, and two features")

    x_c0 = np.array(data[states == 0])
    plt.scatter(x=x_c0[:, 0], y=x_c0[:, 1], c="blue", alpha=0.25, edgecolors="none", label="state 1")
    plt.plot([0, np.mean(x_c0[:, 0])], [0, np.mean(x_c0[:, 1])], color="black")
    plt.plot([np.mean(x_c0[:, 0])], [np.mean(x_c0[:, 1])], marker="o", markersize=8, color="black")

    x_c1 = np.array(data[states == 1])
    plt.scatter(x=x_c1[:, 0], y=x_c1[:, 1], c="orange", alpha=0.25, edgecolors="none", label="state 2")
    plt.plot([0, np.mean(x_c1[:, 0])], [0, np.mean(x_c1[:, 1])], color="black")
    plt.plot([np.mean(x_c1[:, 0])], [np.mean(x_c1[:, 1])], marker="o", markersize=8, color="black", label="mean")

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.show()


# -------------------------------------------- results plots

def _calc_log_marginal(calc_logpmf_mat, data, theta_C, theta_X, **kwargs):
    log_pmf = calc_logpmf_mat(data, theta_C, theta_X, **kwargs)   # shape: (K, M), log_pmf[k, m] = log(p(x[m] | C=k))
    log_likelihood = log_pmf + np.log(theta_C[:, None])           # (K, M), log_likelihood[k, m] = log(p(x[m], C=k))
    log_marginal = LSE(log_likelihood, axis=0)                    # (M, ), log_marginal[m] = log(p(x[m]))
    return log_marginal


def _generate_data_NegativeBinomial_from_parameters(rng, theta_C, r, p, const_seq_depth, m_samples, n_variables, k_states):
    if const_seq_depth is True:
        seq_depth = None
    else:
        seq_depth = rng.binomial(50, 0.5, m_samples)
    data = np.zeros((m_samples, n_variables))
    states = np.zeros(m_samples)
    for m in range(m_samples):
        p_m = p / (p + seq_depth[m] - seq_depth[m] * p) if seq_depth is not None else p
        k = rng.choice(np.arange(k_states), p=theta_C)
        states[m] = k
        data[m] = rng.negative_binomial(n=r[:, k], p=p_m[:, k], size=(1, n_variables))
    return seq_depth, data, states


def full_gmm_normalized_ll(gmm_model: GaussianMixture, data):
    num_samples, num_variables = data.shape
    num_states = gmm_model.means_.shape[0]
    pdf_mat = np.zeros((num_states, num_samples))
    for i in range(num_states):
        epsilon_for_psd = np.quantile(np.abs(gmm_model.covariances_[i]), 0.02) # baseline, using the lowest eiganvalues created poor results
        if epsilon_for_psd < 1e-2:
            epsilon_for_psd = 1e-2
        cov_mat = gmm_model.covariances_[i] + np.eye(num_variables) * epsilon_for_psd
        pdf_mat[i] = mvnorm.logpdf(data, mean=gmm_model.means_[i], cov=cov_mat)
    ll_per_sample = LSE(gmm_model._estimate_log_weights()[:, None] + pdf_mat, axis=0)
    normalized_ll = np.sum(ll_per_sample) / (num_variables * num_samples)
    return normalized_ll


def _clustering_comparison_for_k(rng, r_max, n_EM_runs, const_seq_depth,
                                 num_train_samples, num_test_samples, num_variables, num_states,
                                 train_data, train_seq_depth, test_data, test_seq_depth,
                                 best_theta_C_EM, best_theta_X_EM):
    # sample train & test datasets
    if train_data is None:
        is_simulated_data = True
        real_theta_C = rng.uniform(1, 10, num_states)
        real_theta_C = real_theta_C / real_theta_C.sum()
        real_r = rng.uniform(low=1, high=r_max, size=(num_variables, num_states))
        real_p = rng.uniform(0, 1, (num_variables, num_states))
        train_seq_depth, train_data, _ = _generate_data_NegativeBinomial_from_parameters(rng, real_theta_C, real_r, real_p, const_seq_depth, num_train_samples, num_variables, num_states)
        test_seq_depth, test_data, _ = _generate_data_NegativeBinomial_from_parameters(rng, real_theta_C, real_r, real_p, const_seq_depth, num_test_samples, num_variables, num_states)
    else:
        is_simulated_data = False

    # kmeans - hard EM, and all covariance of all the states is spherical
    normalized_log_marginal_kmeans_train = []
    normalized_log_marginal_kmeans_test = []
    for _ in range(n_EM_runs):
        kmeans = KMeans(n_clusters=num_states, random_state=rng.binomial(100, 0.5)).fit(train_data)
        states_kmeans_train = kmeans.predict(train_data)
        states_kmeans_test = kmeans.predict(test_data)
        centroids = kmeans.cluster_centers_
        cov_kmeans = np.diag(np.full(num_variables, np.var(train_data - centroids[states_kmeans_train])))
        covs_kmeans = np.repeat(cov_kmeans[None, :], num_states, axis=0)
        curr_ll_train = np.sum([multivariate_normal.logpdf(train_data[i],
                                centroids[states_kmeans_train[i]],
                                covs_kmeans[states_kmeans_train[i]]) for i
                                in range(num_train_samples)]) / (num_variables * num_train_samples)
        curr_ll_test = np.sum([multivariate_normal.logpdf(test_data[i],
                               centroids[states_kmeans_test[i]],
                               covs_kmeans[states_kmeans_test[i]]) for i
                               in range(num_test_samples)]) / (num_variables * num_test_samples)
        normalized_log_marginal_kmeans_train.append(curr_ll_train)
        normalized_log_marginal_kmeans_test.append(curr_ll_test)

    # Gaussian Mixture diag - all the states share the same covariance matrix
    normalized_log_marginal_gmix_diag_train = []
    normalized_log_marginal_gmix_diag_test = []
    for _ in range(n_EM_runs):
        gmix_diag: GaussianMixture = GaussianMixture(n_components=num_states, covariance_type="diag", random_state=rng.binomial(100, 0.5)).fit(train_data)
        curr_ll_train = np.sum(gmix_diag.score_samples(train_data)) / (num_variables * num_train_samples)
        curr_ll_test = np.sum(gmix_diag.score_samples(test_data)) / (num_variables * num_test_samples)
        normalized_log_marginal_gmix_diag_train.append(curr_ll_train)
        normalized_log_marginal_gmix_diag_test.append(curr_ll_test)

    # General Gaussian Mixture
    normalized_log_marginal_gmix_train = []
    normalized_log_marginal_gmix_test = []
    for _ in range(n_EM_runs):
        gmix: GaussianMixture = GaussianMixture(n_components=num_states, random_state=rng.binomial(100, 0.5)).fit(train_data)
        curr_ll_train = full_gmm_normalized_ll(gmix, train_data)
        curr_ll_test = full_gmm_normalized_ll(gmix, test_data)
        normalized_log_marginal_gmix_train.append(curr_ll_train)
        normalized_log_marginal_gmix_test.append(curr_ll_test)

    # EM
    normalized_log_marginal_EM_train = []
    normalized_log_marginal_EM_test = []
    normalized_log_marginal_EM_train_seq_depth = []
    normalized_log_marginal_EM_test_seq_depth = []
    EM_inits = ([init_params(num_states=num_states, rng=rng, data=train_data, seq_depth=train_seq_depth, r_max=r_max, init_type="gmm") for _ in range(n_EM_runs)])
    for i in range(n_EM_runs):
        theta_C_init, r_init, p_init = EM_inits[i]
        theta_X_init = np.array([r_init, p_init])
        best_ll, best_theta_C_EM, best_theta_X_EM = -np.inf, theta_C_init, theta_X_init
        try:
            if (is_simulated_data and const_seq_depth) or is_simulated_data is False:
                output_theta_C, output_theta_X, lls = NaiveBayesEM(train_data, theta_C_init, theta_X_init,
                                                                   max_iter=20,
                                                                   print_logs=False,
                                                                   cpd_code=NEGATIVE_BINOMIAL_EM_CODE)
                if lls[-1] > best_ll:
                    best_ll, best_theta_C_EM, best_theta_X_EM = lls[-1], output_theta_C, output_theta_X
                log_marginal_EM_test_per_sample = _calc_log_marginal(calc_logpmf_mat_NegativeBinomial,
                                                                     test_data, output_theta_C, output_theta_X)
                normalized_log_marginal_EM_train.append(lls[-1])
                log_marginal_EM_test = np.sum(log_marginal_EM_test_per_sample) / (num_variables * num_test_samples)
                normalized_log_marginal_EM_test.append(log_marginal_EM_test)
            if (is_simulated_data and not const_seq_depth) or is_simulated_data is False:
                output_theta_C, output_theta_X, lls = NaiveBayesEM(train_data, theta_C_init, theta_X_init, train_seq_depth,
                                                                   rng=rng, max_iter=20, print_logs=False, cpd_code=NEGATIVE_BINOMIAL_SEQ_DEPTH_EM_CODE)
                if lls[-1] > best_ll:
                    best_ll, best_theta_C_EM, best_theta_X_EM = lls[-1], output_theta_C, output_theta_X
                log_marginal_EM_test_seq_depth_per_sample = _calc_log_marginal(calc_logpmf_mat_negative_binomial_sequencing_depth,
                                                                               test_data, output_theta_C, output_theta_X,
                                                                               seq_depth=test_seq_depth)
                normalized_log_marginal_EM_train_seq_depth.append(lls[-1])
                log_marginal_EM_test_seq_depth = np.sum(log_marginal_EM_test_seq_depth_per_sample) / (num_variables * num_test_samples)
                normalized_log_marginal_EM_test_seq_depth.append(log_marginal_EM_test_seq_depth)

        except (AssertionError, ValueError) as exc:
            print(str(exc))
            continue

    ll_kmeans_train_std = np.std(normalized_log_marginal_kmeans_train)
    ll_kmeans_test_std = np.std(normalized_log_marginal_kmeans_test)
    ll_gmix_diag_train_std = np.std(normalized_log_marginal_gmix_diag_train)
    ll_gmix_diag_test_std = np.std(normalized_log_marginal_gmix_diag_test)
    ll_gmix_train_std = np.std(normalized_log_marginal_gmix_train)
    ll_gmix_test_std = np.std(normalized_log_marginal_gmix_test)
    ll_EM_train_std = np.std(normalized_log_marginal_EM_train) if normalized_log_marginal_EM_train else None
    ll_EM_test_std = np.std(normalized_log_marginal_EM_test) if normalized_log_marginal_EM_test else None
    ll_EM_seq_depth_train_std = np.std(normalized_log_marginal_EM_train_seq_depth) if normalized_log_marginal_EM_train_seq_depth else None
    ll_EM_seq_depth_test_std = np.std(normalized_log_marginal_EM_test_seq_depth) if normalized_log_marginal_EM_test_seq_depth else None

    ll_kmeans_train_mean = np.mean(normalized_log_marginal_kmeans_train)
    ll_kmeans_test_mean = np.mean(normalized_log_marginal_kmeans_test)
    ll_gmix_diag_train_mean = np.mean(normalized_log_marginal_gmix_diag_train)
    ll_gmix_diag_test_mean = np.mean(normalized_log_marginal_gmix_diag_test)
    ll_gmix_train_mean = np.mean(normalized_log_marginal_gmix_train)
    ll_gmix_test_mean = np.mean(normalized_log_marginal_gmix_test)
    ll_EM_train_mean = np.mean(normalized_log_marginal_EM_train)
    ll_EM_test_mean = np.mean(normalized_log_marginal_EM_test)
    ll_EM_seq_depth_train_mean = np.mean(normalized_log_marginal_EM_train_seq_depth) if normalized_log_marginal_EM_train_seq_depth else None
    ll_EM_seq_depth_test_mean = np.mean(normalized_log_marginal_EM_test_seq_depth) if normalized_log_marginal_EM_test_seq_depth else None

    return ll_kmeans_train_mean, ll_kmeans_test_mean, \
            ll_gmix_diag_train_mean, ll_gmix_diag_test_mean, \
            ll_gmix_train_mean, ll_gmix_test_mean, \
            ll_EM_train_mean, ll_EM_test_mean, \
            ll_EM_seq_depth_train_mean, ll_EM_seq_depth_test_mean, \
            ll_kmeans_train_std, ll_kmeans_test_std, \
            ll_gmix_diag_train_std, ll_gmix_diag_test_std, \
            ll_gmix_train_std, ll_gmix_test_std, \
            ll_EM_train_std, ll_EM_test_std, \
            ll_EM_seq_depth_train_std, ll_EM_seq_depth_test_std, \
            best_theta_C_EM, best_theta_X_EM


def _clustering_comparison(rng, K_range, r_max, n_EM_runs, const_seq_depth, num_train_samples, num_test_samples, num_variables,
                           train_data, train_seq_depth, test_data, test_seq_depth):
    lls_kmeans_train, lls_kmeans_train_std = [], []
    lls_kmeans_test, lls_kmeans_test_std = [], []
    lls_gmix_diag_train, lls_gmix_diag_train_std = [], []
    lls_gmix_diag_test, lls_gmix_diag_test_std = [], []
    lls_gmix_train, lls_gmix_train_std = [], []
    lls_gmix_test, lls_gmix_test_std = [], []
    lls_EM_train, lls_EM_train_std = [], []
    lls_EM_test, lls_EM_test_std = [], []
    lls_EM_seq_depth_train, lls_EM_seq_depth_train_std = [], []
    lls_EM_seq_depth_test, lls_EM_seq_depth_test_std = [], []

    best_theta_C_EM, best_theta_X_EM = [], [[[] for _ in range(num_variables)], [[] for _ in range(num_variables)]]
    for _K in tqdm(K_range):
        ll_kmeans_train_mean, ll_kmeans_test_mean, \
        ll_gmix_diag_train_mean, ll_gmix_diag_test_mean, \
        ll_gmix_train_mean, ll_gmix_test_mean, \
        ll_EM_train_mean, ll_EM_test_mean, \
        ll_EM_seq_depth_train_mean, ll_EM_seq_depth_test_mean, \
        ll_kmeans_train_std, ll_kmeans_test_std, \
        ll_gmix_diag_train_std, ll_gmix_diag_test_std, \
        ll_gmix_train_std, ll_gmix_test_std, \
        ll_EM_train_std, ll_EM_test_std, \
        ll_EM_seq_depth_train_std, ll_EM_seq_depth_test_std,\
        best_theta_C_EM, best_theta_X_EM = _clustering_comparison_for_k(rng, r_max, n_EM_runs,
                                                                        const_seq_depth,
                                                                        num_train_samples=num_train_samples,
                                                                        num_test_samples=num_test_samples,
                                                                        num_variables=num_variables,
                                                                        num_states=_K,
                                                                        train_data=train_data,
                                                                        train_seq_depth=train_seq_depth,
                                                                        test_data=test_data,
                                                                        test_seq_depth=test_seq_depth,
                                                                        best_theta_C_EM=best_theta_C_EM,
                                                                        best_theta_X_EM=best_theta_X_EM)
        lls_kmeans_train.append(ll_kmeans_train_mean)
        lls_kmeans_test.append(ll_kmeans_test_mean)
        lls_gmix_diag_train.append(ll_gmix_diag_train_mean)
        lls_gmix_diag_test.append(ll_gmix_diag_test_mean)
        lls_gmix_train.append(ll_gmix_train_mean)
        lls_gmix_test.append(ll_gmix_test_mean)
        lls_EM_train.append(ll_EM_train_mean)
        lls_EM_test.append(ll_EM_test_mean)
        if ll_EM_seq_depth_train_mean is not None:
            lls_EM_seq_depth_train.append(ll_EM_seq_depth_train_mean)
        if ll_EM_seq_depth_test_mean is not None:
            lls_EM_seq_depth_test.append(ll_EM_seq_depth_test_mean)

        lls_kmeans_train_std.append(ll_kmeans_train_std)
        lls_kmeans_test_std.append(ll_kmeans_test_std)
        lls_gmix_diag_train_std.append(ll_gmix_diag_train_std)
        lls_gmix_diag_test_std.append(ll_gmix_diag_test_std)
        lls_gmix_train_std.append(ll_gmix_train_std)
        lls_gmix_test_std.append(ll_gmix_test_std)
        lls_EM_train_std.append(ll_EM_train_std)
        lls_EM_test_std.append(ll_EM_test_std)
        if ll_EM_seq_depth_train_std is not None:
            lls_EM_seq_depth_train_std.append(ll_EM_seq_depth_train_std)
        if ll_EM_seq_depth_test_std is not None:
            lls_EM_seq_depth_test_std.append(ll_EM_seq_depth_test_std)

    return lls_kmeans_train, lls_kmeans_train_std, \
        lls_kmeans_test, lls_kmeans_test_std, \
        lls_gmix_diag_train, lls_gmix_diag_train_std, \
        lls_gmix_diag_test, lls_gmix_diag_test_std, \
        lls_gmix_train, lls_gmix_train_std, \
        lls_gmix_test, lls_gmix_test_std, \
        lls_EM_train, lls_EM_train_std, \
        lls_EM_test, lls_EM_test_std, \
        lls_EM_seq_depth_train, lls_EM_seq_depth_train_std, \
        lls_EM_seq_depth_test, lls_EM_seq_depth_test_std


def _plot_clustering_comparison(data, data_part, const_seq_depth, k_range,  # data
                                lls_kmeans_clipped, lls_kmeans_std,  # results
                                lls_gmix_diag_clipped, lls_gmix_diag_std,
                                lls_gmix_clipped, lls_gmix_std,
                                lls_EM, lls_EM_std,
                                lls_EM_seq_depth, lls_EM_seq_depth_std, limits=None):
    # visualization utils
    figsize = (12, 6)
    title_fontsize = 15
    label_fontsize = 10
    margin = 0.5
    large_margin = 4
    fig, axs = plt.subplots(figsize=figsize)
    if data is None:  # simulated data
        if const_seq_depth:
            fig.suptitle(f"normalized log-likelihood of simulated {data_part} data as a function of K", fontsize=title_fontsize)
        else:
            fig.suptitle(f"normalized log-likelihood of simulated {data_part} data with sequencing depth as a function of K",fontsize=title_fontsize)
    else:
        fig.suptitle(f"normalized log-likelihood of real {data_part} data as a function of K", fontsize=title_fontsize)
    axs.set_ylabel("normalized log likelihood", fontsize=label_fontsize)
    axs.set_xlabel("K (number of states)", fontsize=label_fontsize)
    if len(lls_kmeans_clipped) > 0:
        axs.errorbar(k_range, lls_kmeans_clipped, color="pink", ecolor="pink", yerr=lls_kmeans_std, fmt='-o', label="K-Means")
    if len(lls_gmix_diag_clipped) > 0:
        axs.errorbar(k_range, lls_gmix_diag_clipped, color="orange", ecolor="orange", yerr=lls_gmix_diag_std, fmt='-o', label="GMM with diagonal covariance")
    if len(lls_gmix_clipped) > 0:
        axs.errorbar(k_range, lls_gmix_clipped, color="red", ecolor="red", yerr=lls_gmix_std, fmt='-o', label="GMM")
    if len(lls_EM) > 0:
        axs.errorbar(k_range, lls_EM, color="green", ecolor="green", yerr=lls_EM_std, fmt='-o', label="Negative Binomial EM")
    if len(lls_EM_seq_depth) > 0:
        axs.errorbar(k_range, lls_EM_seq_depth, color="blue", ecolor="blue", yerr=lls_EM_seq_depth_std, fmt='-o', label="Negative Binomial EM, Sequencing Depth")
    axs.set_xticks(k_range)
    axs.legend(loc="lower right")
    all_points = np.concatenate((lls_kmeans_clipped, lls_gmix_diag_clipped, lls_gmix_clipped, lls_EM, lls_EM_seq_depth))
    if limits == None:
        upper_limit = np.max(all_points) + margin
        lower_limit = np.min(all_points) - margin
    else:
        upper_limit = limits[0]
        lower_limit = limits[1]
    axs.set_ylim([lower_limit, upper_limit])
    return fig, axs


def plot_clustering_comparison(rng, K_max, r_max=99, n_EM_runs=5, const_seq_depth=True, num_train_samples=M, num_test_samples=M,
                               num_variables=N,
                               train_data=None, train_seq_depth=None, test_data=None, test_seq_depth=None, force=False):
    k_range = np.arange(1, K_max + 1)
    if train_data is not None:
        num_train_samples, num_variables = train_data.shape
        num_test_samples, _ = test_data.shape

    if not force:
        train_files = np.load("plots/clustering_likelihood_train_M_49_N_30_k_max_10_real_data/data_file.npz")
        lls_kmeans_train, lls_kmeans_train_std, \
        lls_gmix_diag_train,lls_gmix_diag_train_std, \
        lls_gmix_train, lls_gmix_train_std, \
        lls_EM_train, lls_EM_train_std, \
        lls_EM_seq_depth_train, lls_EM_seq_depth_train_std = [train_files[f] for f in train_files.files]

        test_files = np.load("plots/clustering_likelihood_test_M_12_N_30_k_max_10_real_data/data_file.npz")
        lls_kmeans_test, lls_kmeans_test_std, \
        lls_gmix_diag_test, lls_gmix_diag_test_std, \
        lls_gmix_test, lls_gmix_test_std, \
        lls_EM_test, lls_EM_test_std, \
        lls_EM_seq_depth_test, lls_EM_seq_depth_test_std = [test_files[f] for f in test_files.files]
    else:
        lls_kmeans_train, lls_kmeans_train_std, \
        lls_kmeans_test, lls_kmeans_test_std, \
        lls_gmix_diag_train, lls_gmix_diag_train_std, \
        lls_gmix_diag_test, lls_gmix_diag_test_std, \
        lls_gmix_train, lls_gmix_train_std, \
        lls_gmix_test, lls_gmix_test_std, \
        lls_EM_train, lls_EM_train_std, \
        lls_EM_test, lls_EM_test_std, \
        lls_EM_seq_depth_train, lls_EM_seq_depth_train_std, \
        lls_EM_seq_depth_test, lls_EM_seq_depth_test_std = _clustering_comparison(rng, k_range, r_max, n_EM_runs,
                                                                                      const_seq_depth,
                                                                                      num_train_samples=num_train_samples,
                                                                                      num_test_samples=num_test_samples,
                                                                                      num_variables=num_variables,
                                                                                      train_data=train_data,
                                                                                      train_seq_depth=train_seq_depth,
                                                                                      test_data=test_data,
                                                                                      test_seq_depth=test_seq_depth)


    # plot without clipping
    fig_train, axs_train = _plot_clustering_comparison(train_data, "train",
                                                       const_seq_depth, k_range,
                                                       lls_kmeans_train, lls_kmeans_train_std,
                                                       lls_gmix_diag_train, lls_gmix_diag_train_std,
                                                       lls_gmix_train, lls_gmix_train_std,
                                                       lls_EM_train, lls_EM_train_std,
                                                       lls_EM_seq_depth_train, lls_EM_seq_depth_train_std)
    if train_data is None:  # simulated data
        fig_test, axs_test = _plot_clustering_comparison(test_data, "test",
                                                         const_seq_depth,
                                                         k_range,
                                                         lls_kmeans_test, lls_kmeans_test_std,
                                                         lls_gmix_diag_test, lls_gmix_diag_test_std,
                                                         lls_gmix_test, lls_gmix_test_std,
                                                         lls_EM_test, lls_EM_test_std,
                                                         lls_EM_seq_depth_test, lls_EM_seq_depth_test_std)
        if const_seq_depth:
            train_figname = f"clustering_likelihood_train_r_max_{r_max}_M_{num_train_samples}_N_{num_variables}_k_max_{K_max}"
            test_figname = f"clustering_likelihood_test_r_max_{r_max}_M_{num_test_samples}_N_{num_variables}_k_max_{K_max}"
        else:
            train_figname = f"clustering_likelihood_train_r_max_{r_max}_M_{num_train_samples}_N_{num_variables}_k_max_{K_max}_seq_depth"
            test_figname = f"clustering_likelihood_test_r_max_{r_max}_M_{num_test_samples}_N_{num_variables}_k_max_{K_max}_seq_depth"
    else:  # real_data
        fig_test, axs_test = _plot_clustering_comparison(test_data, "test",
                                                         const_seq_depth, k_range,
                                                         lls_kmeans_test, lls_kmeans_test_std,
                                                         lls_gmix_diag_test, lls_gmix_diag_test_std,
                                                         lls_gmix_test, lls_gmix_test_std,
                                                         lls_EM_test, lls_EM_test_std,
                                                         lls_EM_seq_depth_test, lls_EM_seq_depth_test_std,
                                                         limits=(-5, -12))
        gmm_fig_test, gmm_axs_test = _plot_clustering_comparison(test_data, "test",
                                                                 const_seq_depth, k_range,
                                                                 [], [], [], [],
                                                                 lls_gmix_test, lls_gmix_test_std,
                                                                 [], [], [], [],
                                                                 limits=(-12, -210))
        train_figname = f"clustering_likelihood_train_M_{num_train_samples}_N_{num_variables}_k_max_{K_max}_real_data"
        test_figname = f"clustering_likelihood_test_M_{num_test_samples}_N_{num_variables}_k_max_{K_max}_real_data"

    # save figs and data
    if force:
        savefig_and_data(fig_train, "plots", train_figname, lls_kmeans_train,
                         lls_kmeans_train_std,
                         lls_gmix_diag_train, lls_gmix_diag_train_std,
                         lls_gmix_train, lls_gmix_train_std, lls_EM_train,
                         lls_EM_train_std, lls_EM_seq_depth_train,
                         lls_EM_seq_depth_train_std)
        savefig_and_data(fig_test, "plots", test_figname, lls_kmeans_test,
                         lls_kmeans_test_std,
                         lls_gmix_diag_test, lls_gmix_diag_test_std, lls_gmix_test,
                         lls_gmix_test_std, lls_EM_test,
                         lls_EM_test_std, lls_EM_seq_depth_test,
                         lls_EM_seq_depth_test_std)
    else:
        work_folder = f"plots/{train_figname}"
        fig_train.savefig(fname=f"{work_folder}/{train_figname}")
        work_folder = f"plots/{test_figname}"
        fig_test.savefig(fname=f"{work_folder}/{test_figname}")

    if train_data is not None:  # real data
        work_folder = f"plots/{test_figname}"
        gmm_fig_test.savefig(
            fname=f"{work_folder}/gmm_loglikelihood_test_M_{num_test_samples}_N_{num_variables}_k_max_{K_max}_real_data")
    plt.show()


def bic_plot(rng, data, seq_depth, k_arr, fig_name, cpd_code):
    M, N = data.shape
    bic_list = []
    best_theta_c, best_theta_x = None, None
    k_len = k_arr.shape[0]
    best_ll_arr = np.zeros(k_arr.shape[0])
    for i in range(k_len):
        k = k_arr[i]
        best_ll = [-np.inf]
        if k > 1:
            theta_C_init, theta_X_init = init_param_old_em(rng, best_theta_c, best_theta_x, N)
            theta_C, theta_X, likelihoods = NaiveBayesEM(data, theta_C_init, theta_X_init,
                                                         seq_depth, cpd_code=cpd_code,
                                                         rng=rng, max_iter=15)
            if best_ll[-1] < likelihoods[-1]:
                best_ll = likelihoods
                best_theta_x = theta_X
                best_theta_c = theta_C
        theta_C_init, r_init, p_init = init_params(rng, data, seq_depth, num_states=k)
        theta_X_init = np.array([r_init, p_init])

        theta_C, theta_X, likelihoods = NaiveBayesEM(data, theta_C_init, theta_X_init, seq_depth,
                                                     rng=rng, max_iter=15, cpd_code=cpd_code)
        if best_ll[-1] < likelihoods[-1]:
            best_ll = likelihoods
            best_theta_x = theta_X
            best_theta_c = theta_C
        model_ll = best_ll[-1]
        free_parameters = free_parameters_NB(k, N)
        bic_list.append(bic_score(model_ll * M * N, free_parameters, M))
        best_ll_arr[i] = model_ll
        print(bic_list[-1])
        print(model_ll)

    plt.plot(k_arr, bic_list)
    plt.xlabel("Number of Parameters")
    plt.ylabel("BIC")
    plt.title("BIC as Function of the Model Parameters")
    # plt.savefig(fname=f"plots/bic_plot_N{N}_K{K}_M{M}_sim_data_with_seq_depth")
    savefig_and_data(plt, "plots/", f"bic_plot_N{N}_K{K}_M{M}_{fig_name}", np.array(bic_list), best_ll_arr)
    plt.show()


def plot_category_accuracy(train_results, test_results, categories, title, plot_folder, fig_name):
    plt.figure(figsize=(8, 6))
    bar_width = 0.35
    x = np.arange(len(categories))
    plt.bar(x - bar_width / 2, train_results, bar_width, label='train data')
    plt.bar(x + bar_width / 2, test_results, bar_width, label='test data')
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title(title)
    tick_labels = ['\n'.join(category.split()) for category in categories]
    plt.xticks(x, tick_labels)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.tight_layout()
    savefig_and_data(plt, plot_folder, fig_name, train_results, test_results)
    plt.show()


def accuracy_plot(rng, train_data, train_seq_depth, test_data, test_seq_depth, train_tags, test_tags,
                  n_EM_runs=1, num_states=3, calc_logpmf_mat=calc_logpmf_mat_negative_binomial_sequencing_depth):
    num_train_samples, num_variables = train_data.shape
    num_test_samples, _ = test_data.shape
    inverse_map, states_train = np.unique(train_tags, return_inverse=True)
    inverse_dict = dict(zip(inverse_map, range(num_states)))
    states_test = np.array([inverse_dict[tag] for tag in test_tags])

    # ---------- run EM with sequencing depth
    best_train_ll = -np.inf
    output_theta_C, output_theta_X = None, None
    for _ in range(n_EM_runs):
        theta_C_init, r_init, p_init = init_params(rng, train_data, train_seq_depth,
                                                   num_variables=num_variables, num_states=num_states,
                                                   r_max=np.max(train_data), init_type="gmm")
        theta_X_init = np.array([r_init, p_init])
        output_theta_C_tag, output_theta_X_tag, lls = NaiveBayesEM(train_data, theta_C_init, theta_X_init,
                                                                   train_seq_depth,
                                                                   rng=rng, max_iter=20, print_logs=False)
        if lls[-1] > best_train_ll:
            output_theta_C, output_theta_X = output_theta_C_tag, output_theta_X_tag

    _, train_log_posterior = calc_log_marginal_and_posterior(calc_logpmf_mat, train_data, output_theta_C, output_theta_X,
                                                             seq_depth=train_seq_depth)
    _, test_log_posterior = calc_log_marginal_and_posterior(calc_logpmf_mat, test_data, output_theta_C, output_theta_X,
                                                            seq_depth=test_seq_depth)
    # ---------- find the best permutation for EM
    EM_train_accuracy = 0
    EM_test_accuracy = 0
    for perm in list(permutations(range(num_states))):
        perm_train_accuracy = calc_accuracy_for_perm(train_log_posterior, states_train, perm)
        if perm_train_accuracy > EM_train_accuracy:
            EM_train_accuracy = perm_train_accuracy
            EM_test_accuracy = calc_accuracy_for_perm(test_log_posterior, states_test, perm)

    # ---------- kmeans
    kmeans: KMeans = KMeans(n_clusters=num_states, random_state=rng.binomial(100, 0.5)).fit(train_data)
    kmeans_states_train = kmeans.predict(train_data)
    kmeans_train_accuracy = 0
    kmeans_test_accuracy = 0
    for perm in list(permutations(range(3))):
        perm_train_accuracy = np.mean(np.array(perm)[kmeans_states_train] == states_train)
        if perm_train_accuracy > kmeans_train_accuracy:
            kmeans_train_accuracy = perm_train_accuracy
            kmeans_states_test = kmeans.predict(test_data)
            kmeans_test_accuracy = np.mean(np.array(perm)[kmeans_states_test] == states_test)

    # ---------- kmeans with standardization
    standardized_train_data = train_data / train_seq_depth[:, None]
    standardized_test_data = test_data / test_seq_depth[:, None]
    kmeans_norm: KMeans = KMeans(n_clusters=num_states, random_state=rng.binomial(100, 0.5)).fit(standardized_train_data)
    standardized_kmeans_states_train = kmeans_norm.predict(standardized_train_data)
    standardized_kmeans_train_accuracy = 0
    standardized_kmeans_test_accuracy = 0
    for perm in list(permutations(range(3))):
        perm_train_accuracy = np.mean(np.array(perm)[standardized_kmeans_states_train] == states_train)
        if perm_train_accuracy > standardized_kmeans_train_accuracy:
            standardized_kmeans_train_accuracy = perm_train_accuracy
            standardized_kmeans_states_test = kmeans_norm.predict(standardized_test_data)
            standardized_kmeans_test_accuracy = np.mean(np.array(perm)[standardized_kmeans_states_test] == states_test)

    # ---------- complete dataset
    full_train_data, _, _, full_test_data, _, _, _ =\
        load_processed_real_data(execute_feature_selection=False)
    kmeans_full: KMeans = KMeans(n_clusters=3, random_state=rng.binomial(100, 0.5)).fit(full_train_data.values)
    kmeans_states_full_train = kmeans_full.predict(full_train_data)
    kmeans_full_train_accuracy = 0
    kmeans_full_test_accuracy = 0
    for perm in list(permutations(range(3))):
        perm_train_accuracy = np.mean(np.array(perm)[kmeans_states_full_train] == states_train)
        if perm_train_accuracy > kmeans_full_train_accuracy:
            kmeans_full_train_accuracy = perm_train_accuracy
            kmeans_states_full_test = kmeans_full.predict(full_test_data)
            kmeans_full_test_accuracy = np.mean(np.array(perm)[kmeans_states_full_test] == states_test)

    standardized_full_train_data = full_train_data / train_seq_depth[:, None]
    standardized_full_test_data = full_test_data / test_seq_depth[:, None]
    kmeans_full_norm: KMeans = KMeans(n_clusters=3, random_state=rng.binomial(100, 0.5)).fit(standardized_full_train_data)
    standardized_kmeans_states_full_train = kmeans_full_norm.predict(standardized_full_train_data)
    standardized_kmeans_full_train_accuracy = 0
    standardized_kmeans_full_test_accuracy = 0
    for perm in list(permutations(range(3))):
        perm_train_accuracy = np.mean(np.array(perm)[standardized_kmeans_states_full_train] == states_train)
        if perm_train_accuracy > standardized_kmeans_full_train_accuracy:
            standardized_kmeans_full_train_accuracy = perm_train_accuracy
            standardized_kmeans_states_full_test = kmeans_full_norm.predict(standardized_full_test_data)
            standardized_kmeans_full_test_accuracy = np.mean(np.array(perm)[standardized_kmeans_states_full_test] == states_test)

    train_results = [kmeans_full_train_accuracy, standardized_kmeans_full_train_accuracy, kmeans_train_accuracy,
                     standardized_kmeans_train_accuracy, EM_train_accuracy]
    test_results = [kmeans_full_test_accuracy, standardized_kmeans_full_test_accuracy, kmeans_test_accuracy,
                    standardized_kmeans_test_accuracy, EM_test_accuracy]

    # files = np.load(f"plots/accuracy_plot_over_expression_only_{OVER_EXPRESSED_ONLY}/data_file.npz")
    # train_results, test_results = [files[f] for f in files.files]

    # ---------- results
    categories = ['kmeans all genes', 'kmeans all genes after standardizing',
                  'kmeans after feature selection', 'kmeans after feature selection and standardizing',
                  'Negative Binomial EM']

    print("KMeans - all the genes")
    print(f"\tbest accuracy on train - {train_results[0]}")
    print(f"\ttest accuracy - {test_results[0]}")
    print("KMeans - all the genes and after standardized data with sequencing depth")
    print(f"\tbest accuracy on train - {train_results[1]}")
    print(f"\ttest accuracy - {test_results[1]}")
    print("KMeans - after feature selection")
    print(f"\tbest accuracy on train - {train_results[2]}")
    print(f"\ttest accuracy - {test_results[2]}")
    print("KMeans - after standardized data with sequencing depth and feature selection")
    print(f"\tbest accuracy on train - {train_results[3]}")
    print(f"\ttest accuracy - {test_results[3]}")
    print("Negative Binomial EM - after feature selection")
    print(f"\tbest accuracy on train - {train_results[4]}")
    print(f"\ttest accuracy - {test_results[4]}")

    plot_title = 'Accuracy of 3 cell types for different clustering algorithms with 3 clusters'
    fig_name = f"accuracy_plot_over_expression_only_{OVER_EXPRESSED_ONLY}"
    plot_category_accuracy(train_results, test_results, categories, plot_title, "plots/", fig_name)


def plot_train_test(k_list, train_ll, test_ll, plot_folder, figname):
    plt.plot(k_list, train_ll, label="train")
    plt.plot(k_list, test_ll, label="test")
    plt.title("train-test log likelihood")
    plt.xlabel("K")
    plt.ylabel("LL")
    plt.legend()
    print(f"{plot_folder}/{figname}")
    plt.show()


def run_train_test(rng, train_data, test_data, train_seq_depth, test_seq_depth, k_list, iter_num, cpd_code, plot_folder, figname):
    num_train_samples, num_variables = train_data.shape
    num_test_samples, _ = test_data.shape
    log_pmf_function = EM_COMPONENTS[cpd_code][3]
    train_ll_list = []
    test_ll_list = []
    best_theta_c, best_theta_x = None, None
    for k in k_list:
        best_ll = -np.Inf
        for i in range(iter_num):
            theta_C_init, r_init, p_init = init_params(rng, data=train_data.values, seq_depth=train_seq_depth, num_variables=num_variables, num_states=k)
            theta_C, theta_X, likelihoods = NaiveBayesEM(train_data.values, theta_C_init, np.array([r_init, p_init]), train_seq_depth, rng, max_iter=15, cpd_code=cpd_code)
            if likelihoods[-1] > best_ll:
                best_ll = likelihoods[-1]
                best_theta_c = theta_C
                best_theta_x = theta_X
        train_ll_list.append(best_ll)
        test_ll = _calc_log_marginal(log_pmf_function, test_data, best_theta_c, best_theta_x)
        test_ll_normalize = np.sum(test_ll) / (num_variables * num_test_samples)
        test_ll_list.append(test_ll_normalize)

    plot_train_test(k_list, train_ll_list, test_ll_list, plot_folder, figname)


def load_cluster_data(data_path):
    # arr0,1 - kmeans arr, arr2,3 - gmm_diag arr4,5 - gmm_full
    # arr6,7 - em, arr 8,9 - em_with_seq
    cluster_data = np.load(data_path)
    data_dict = {"lls_kmeans": cluster_data["arr_0"], "lls_kmeans_std": cluster_data["arr_1"],
                 "lls_gmix_diag": cluster_data["arr_2"], "lls_gmix_diag_std": cluster_data["arr_3"],
                 "lls_gmix": cluster_data["arr_4"], "lls_gmix_std": cluster_data["arr_5"],
                 "lls_EM": cluster_data["arr_6"], "lls_EM_std": cluster_data["arr_7"],
                 "lls_EM_seq_depth": cluster_data["arr_8"], "lls_EM_seq_depth_std": cluster_data["arr_9"]}
    return data_dict


def save_new_data(work_folder, filename, *args):
    np.savez(f"{work_folder}/{filename}", *args)


# -------------------------------------------- Main

def figure3(train_data, selected_genes_names, train_seq_depth, train_tags):
    # heat map of the data
    log_data = np.log(train_data.values / train_seq_depth[:, None] + 1)
    df = pd.DataFrame(log_data, columns=selected_genes_names, index=train_tags)
    title = f"Heatmap of 30 Genes"
    data_heatmap = sns.clustermap(df, row_cluster=True, figsize=(10, 8),
                                  cbar_pos=(0.08, 0.2, 0.03, 0.5), yticklabels=True)
    data_heatmap.ax_row_dendrogram.set_visible(False)
    data_heatmap.ax_col_dendrogram.set_visible(False)
    data_heatmap.ax_heatmap.set_title(title, fontsize=16)
    data_heatmap.ax_cbar.set_ylabel("Log(Normalized Data)")
    savefig_and_data(plt, "plots/",
                     f"cluster_map_feature_selection_over_expression_only_{OVER_EXPRESSED_ONLY}",
                     df)
    plt.show()


def figure4ab(rng, data, states, include_init,
            true_p, seq_depth, cpd_code, n_runs, n_iters):
    # 1. random init
    # 2. random init with the same parameters for each cluster without uniform noise
    plot_p_per_iteration(rng, data, states, include_init, True, True,
            true_p, seq_depth, cpd_code, n_runs, n_iters)


def figure4_5(rng, data, seq_depth, states, true_theta_C, true_r, true_p):
    # 1. parameter plot with random init with uniform noise and one gmm init
    # 2. likelihood with 10 random init with uniform with one gmm init
    plot_parameters_and_likelihood_per_iteration(rng, data, seq_depth, states,
                                                 true_theta_C, true_r, true_p,
                                                 n_iters=10, n_runs=11)


def figure6(rng, real_train_data, real_train_seq_depth,
                  real_test_data, real_test_seq_depth,
                  train_tags, test_tags):
    # accuracy to cell types
    accuracy_plot(rng, real_train_data.values, real_train_seq_depth,
                  real_test_data.values, real_test_seq_depth,
                  train_tags, test_tags)


def figure7(rng):
    # cluster comparision with simulated data
    plot_clustering_comparison(rng, K_max=10, r_max=5, n_EM_runs=5,
                               num_train_samples=5000, num_test_samples=5000,
                               num_variables=30)


def figure8(rng, real_train_data, real_train_seq_depth, real_test_data, real_test_seq_depth, force=False):
    # cluster comparision with real data
    plot_clustering_comparison(rng, K_max=10, n_EM_runs=3,
                               train_data=real_train_data.values, train_seq_depth=real_train_seq_depth,
                               test_data=real_test_data.values, test_seq_depth=real_test_seq_depth, force=force)


def figure9(rng, simulated_data, simulated_seq_depth):
    # bic on simulated data
    bic_plot(rng, simulated_data, simulated_seq_depth, np.arange(1, 10), "sim_data", 1)
    bic_plot(rng, simulated_data, simulated_seq_depth, np.arange(1, 10), "sim_data_with_seq_depth", 2)


def figure10(rng, real_train_data, real_train_seq_depth):
    # bic on real data with seq_depth and without
    bic_plot(rng, real_train_data.values, real_train_seq_depth,
             np.arange(1, 10), "real_data", 1)
    bic_plot(rng, real_train_data.values, real_train_seq_depth,
             np.arange(1, 10), "real_data_with_seq_depth", 2)


def figure_supp(rng, m_list, num_runs, num_iter, n_list, seq_depth):
    # 1. plot of likelihood as function of M
    # 2. plot of likelihood as function of N
    # 3. plot of likelihood as function of K # MISSING

    plot_likelihood_as_func_of_M(rng, m_list, num_runs, num_iter)
    plot_likelihood_as_func_of_N(rng, n_list, seq_depth)


def run_plots(seed):
    rng = np.random.default_rng(seed)
    simulated_theta_C, simulated_r, simulated_p, simulated_seq_depth, simulated_data, states = \
        generate_data_NegativeBinomial(rng=rng)
    real_train_data, real_train_seq_depth, train_tags, real_test_data, \
    real_test_seq_depth, test_tags, selected_genes_names = \
        load_processed_real_data(DATA_LOCATION)

    # -------------------------- data heatmap plot
    figure3(real_train_data, selected_genes_names, real_train_seq_depth, train_tags)

    # -------------------------- parameters and likelihood convergence
    figure4_5(rng, simulated_data, states, simulated_theta_C,
              simulated_r, simulated_p, simulated_seq_depth)

    # -------------------------- accuracy to cell types
    figure6(rng, real_train_data, real_train_seq_depth,
                      real_test_data, real_test_seq_depth,
                      train_tags, test_tags)

    # -------------------------- clustering comparison
    figure7(rng)  # simulated data
    figure8(rng, real_train_data, real_train_seq_depth,
            real_test_data, real_test_seq_depth)  # real data

    # -------------------------- BIC plot
    figure9(rng, simulated_data, simulated_seq_depth)  # simulated data
    figure10(rng, real_train_data, real_train_seq_depth)  # real data

    # --------------------- END PLOTS -------------------


if __name__ == "__main__":
    SEED = 60
    run_plots(SEED)


# -------------------------------------------- Archived

# m_list = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000])
# n_list = np.arange(4, 21, 4)
# figure_supp(rng, m_list, 10, 5, n_list, simulated_seq_depth)


# def theta_c_distribution_plot(theta_c_arr, K):
#     for k in range(K):
#         plt.hist(theta_c_arr[:, k], alpha=0.5, label=f'{k}')
#     plt.legend(loc='upper right')
#     plt.show()


# def old_heatmap(normalize_data, predictions, NUM_CLUSTERS, selected_genes_ind, selected_genes_names, NUM_FROM_EACH_CLUSTER):
#     means = np.array([np.mean(normalize_data[predictions == c], axis=0) for c in
#                       range(NUM_CLUSTERS)])
#     selected_means = means[:, selected_genes_ind]
#     normalized_means = selected_means / np.sum(selected_means, axis=0)
#     df = pd.DataFrame(normalized_means, columns=selected_genes_names)
#     title = f"A cluster map of the {NUM_FROM_EACH_CLUSTER} most significant genes of each cluster"
#     cluster = sns.clustermap(df, method="single", row_cluster=False, figsize=(10, 8), cbar_pos=(0.08, 0.2, 0.03, 0.5))
#     cluster.fig.suptitle(title, fontsize=16)
#     savefig_and_data(plt, "plots/", f"cluster_map_feature_selection_over_expression_only_{OVER_EXPRESSED_ONLY}", df)
#     plt.show()

# def r_cpd_distribution(r_arr, i, k):
#     r_vec = r_arr[:, i, k]
#     r_vec = r_vec[r_vec < 100]
#     plt.hist(r_vec)
#     plt.show()


# def real_data_plots(seed, n_iter):
#     rng = np.random.default_rng(seed)
#     data_folder = "/Users/shakedamar/PycharmProjects/final_project/Data/"
#     all_data = load_real_data(data_folder + "data.csv", data_folder + "seq_depth.csv",
#                               data_folder + "tags.csv")
#     data, seq_depth, tags = all_data
#     b, states = np.unique(tags, return_inverse=True)
#     data = find_significant_genes(data, 20)
#     K = np.unique(states).shape[0]
#     M, N = data.shape
#
#     theta_c_arr = np.zeros((n_iter, K))
#     r_arr = np.zeros((n_iter, N, K))
#
#     for i in range(n_iter):
#         theta_C_init, r_init, p_init = init_params(data, rng=rng,
#                                                    uniform_noise=UNIFORM_NOISE_INIT,
#                                                    num_variables=N,
#                                                    num_states=K)
#         theta_X_init = np.array([r_init, p_init])
#         output_theta_C, output_theta_X, lls = \
#         Naive_Bayes_EM(data, theta_C_init, theta_X_init, seq_depth,
#                        rng=rng, max_iter=15)
#         sorted_theta_c = output_theta_C[output_theta_C.argsort()]
#         theta_c_arr[i] = sorted_theta_c
#         output_r = output_theta_X[0]
#         r_arr[i] = output_r[:, output_theta_C.argsort()]
#     # theta_c_distribution_plot(theta_c_arr, K)
#     r_cpd_distribution(r_arr, 0, 0)
#     # accuaracy - with permutation - input data, states
#     accuracy_plot(rng, data, seq_depth, tags)
#
#     # bic_plot - input data
#     # bic_plot(rng, data, seq_depth, k_list=np.arange(1, 11))


# def test_r(seed, n_iter):
#     rng = np.random.default_rng(seed)
#     theta_C, r, p, seq_depth, data, states = generate_easy_data(rng)
#     M, N = data.shape
#     theta_c_arr = np.zeros((n_iter, K))
#     r_arr = np.zeros((n_iter, N, K))
#
#     for i in range(n_iter):
#         theta_C_init, r_init, p_init = init_params(rng, uniform_noise=UNIFORM_NOISE_INIT,
#                                                    n_variables=N, k_states=K)
#         theta_X_init = np.array([r_init, p_init])
#         output_theta_C, output_theta_X, lls = \
#             Naive_Bayes_EM(data, theta_C_init, theta_X_init, seq_depth,
#                            rng=rng, max_iter=15)
#         output_r = output_theta_X[0]
#         theta_C_MLE, r_MLE, p_MLE = compute_empirical_MLE_NegativeBinomial(data, states, K, rng=rng)
#         orig, perm = linear_sum_assignment(- r_MLE.T @ output_r)
#         sorted_theta_c = output_theta_C[perm]
#         theta_c_arr[i] = sorted_theta_c
#         r_arr[i] = output_r[:, perm]
#     # theta_c_distribution_plot(theta_c_arr, K)
#     r_cpd_distribution(r_arr, 0, 0)
# -------------------------- train-test plot
# run_train_test(rng, real_train_data, real_test_data,
#                real_train_seq_depth, real_test_seq_depth,
#                np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45]), iter_num=5, cpd_code=1, plot_folder="plots/", figname=f"train_test_plot_N{N}_K{K}_M{M}_real_data")
# run_train_test(rng, real_train_data, real_test_data,
#                real_train_seq_depth, real_test_seq_depth,
#                np.arange(1, 10), iter_num=5, cpd_code=2,
#                plot_folder="plots/",
#                figname=f"train_test_plot_N{N}_K{K}_M{M}_real_data_with_seq_depth")
