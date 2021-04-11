from proNE import *

def run(graph, emb1, emb2, dim, step, mu, theta):
    print("Running", graph, dim, step, mu, theta)
    t_0 = time.time()
    model = ProNE(graph, emb1, emb2, dim)
    t_1 = time.time()

    features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
    t_2 = time.time()

    embeddings_matrix = model.chebyshev_gaussian(
        model.matrix0, features_matrix, step, mu, theta)
    t_3 = time.time()

    print('---', model.node_number)
    print('total time', t_3 - t_0)
    print('sparse NE time', t_2 - t_1)
    print('spectral Pro time', t_3 - t_2)

    save_embedding(emb1, features_matrix)
    save_embedding(emb2, embeddings_matrix)
    print('save embedding done')


def main(task, suffix):
    file_name = task if suffix == "" else "{}_{}".format(task, suffix)
    graph = "../graph/{}/{}.edgelist".format(task, file_name)

    for step in [10, 20]:
        for mu in [0.1, 0.2, 0.5, 1]:
            for theta in [0.1, 0.2, 0.5, 1]:
                for dim in [5, 20, 50, 100, 200]:
                    s = "{}_{}_{}_{}".format(step, mu, theta, dim)
                    emb1 = "./emb/{}_sparse.emb{}".format(file_name, s)
                    emb2 = "./emb/{}_spectral.emb{}".format(file_name, s)
                    run(graph, emb1, emb2, dim, step, mu, theta)


main("ftp", "")
main("ncaa", "")