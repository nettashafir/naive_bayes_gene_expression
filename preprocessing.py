from utils import *


NUM_CLUSTERS = 3
NUM_FROM_EACH_CLUSTER = 10


def main(seed):
    rng = np.random.default_rng(seed)

    # test_train_split(data_location)
    train_data = pd.read_csv(DATA_LOCATION + "train.csv")
    train_seq_depth = np.loadtxt(DATA_LOCATION + "train_seq_depth.csv", delimiter=",")
    train_tags = []
    with open(DATA_LOCATION + "train_tags.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                train_tags.append(row[0])
    b, states = np.unique(train_tags, return_inverse=True)
    normalize_data = train_data.values / train_seq_depth[:, None]

    # ------------------------- find the significance genes, and rank them
    significant_genes_per_cluster, predictions = find_significant_genes(normalize_data, NUM_CLUSTERS, rng)

    len1 = len(significant_genes_per_cluster[0])
    len2 = len(significant_genes_per_cluster[1])
    len3 = len(significant_genes_per_cluster[2])
    print(len1)
    print(len2)
    print(len3)
    print(len1+len3+len2)

    # ------------------------- feature selection - take the most significant genes from each cluster
    selected_genes_ind = np.concatenate([np.array(cluster)[:NUM_FROM_EACH_CLUSTER, 0] for cluster in
                                         significant_genes_per_cluster]).astype(int)
    # selected_genes_names = train_data.columns[selected_genes_ind].tolist()

    selected_genes_names = []
    for c in range(NUM_CLUSTERS):
        names = train_data.columns[[np.array(significant_genes_per_cluster[c])[:, 0].astype(int)]].tolist()
        with open(DATA_LOCATION + f"significant_genes_cluster_{c}_over_exp_only_{OVER_EXPRESSED_ONLY}.txt", mode="w") as f:
            f.write("\n".join(names))
        selected_genes_names.extend(names[:NUM_FROM_EACH_CLUSTER])

    filename = f"selected_genes_over_exp_only_{OVER_EXPRESSED_ONLY}.csv"
    # np.savetxt(DATA_LOCATION + filename, selected_genes_names, delimiter=",", fmt="%s")



if __name__ == "__main__":
    seed = 42
    main(seed)
