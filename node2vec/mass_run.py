from src/main import *

def run(weighted, num_walks, walk_length, input, output):
	nx_G = read_graph()
	print("Reading Done!")
	G = node2vec.Graph(nx_G, 1, 1, weighted)
	print("Creation Done!")
	G.preprocess_transition_probs()
	print("Preprocess Done!")
	walks = G.simulate_walks(num_walks, walk_length)
	print("Walking Done!")
	walks_save_path = "walks/" + input.split("/")[-1] + "_" +  ".txt" 
	with open(walks_save_path, 'w') as f:
		for walk in walks: 
			f.writelines("%s " % place for place in walk)
			f.writelines("\n")

	learn_embeddings(walks)

	cnts = pd.DataFrame(walks).stack().value_counts()
	restart_lst = list(cnts[cnts < cnts.quantile(0.25)].index)
	additional_walks = max(int(num_walks * 0.1), 4)
	print("additional walks", additional_walks)
	restart_walks = G.simulate_walks(additional_walks * 4, walk_length, nodes=restart_lst)
	output = output[:-4] + "_restart.emb"

	walks_save_path = "walks/" + input.split("/")[-1] + "_restart.txt"
	new_walks = restart_walks + walks[:-additional_walks * cnts.shape[0]]
	with open(walks_save_path, 'w') as f:
		for walk in new_walks: 
			f.writelines("%s " % place for place in walk)
			f.writelines("\n")
	learn_embeddings(new_walks)


def main(task, suffix):
	file_name = task if suffix == "" else "{}_{}".format(task, suffix)
	input = "../graph/{}/{}.edgelist".format(task, file_name)


    for weighted in [False]:
                    s = "{}_{}_{}_{}".format(step, mu, theta, dim)
                    output = "./emb/{}.emb".format(file_name)
                    run(weighted, num_walks, walk_length, input, output) 


main("genes", "")