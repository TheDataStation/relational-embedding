import csv
import pandas as pd

name_basics_file = "../data/imdb/name.basics.tsv"

def obtain_ground_truth_name(sample_size = 1000):
    df = pd.read_csv(name_basics_file, encoding = 'latin1', sep = '\t')
    sample = df.sample(n = sample_size)
    sample_query = sample.drop(columns = ['primaryName']).values.tolist()
    ground_truth = sample['primaryName'].values.tolist()
    return sample_query, ground_truth 

def obtain_ground_truth_profession(sample_size = 1000):
    df = pd.read_csv(name_basics_file, encoding = 'latin1', sep = '\t')
    sample = df.sample(n = sample_size)
    sample_query = sample.drop(columns = ['primaryProfession']).values.tolist()
    ground_truth = sample['primaryProfession'].values.tolist()
    return sample_query, ground_truth 

def measure_quality(ground_truth, predicted_truth):
    precision = [] 
    for i in range(len(ground_truth)):
        flag = ground_truth[i] in predicted_truth[i]
        precision.append(flag)
    return precision

if __name__ == "__main__":
    print("Evaluation Utils:")

    # test 1 : obtain grouth truth for missing names 
    print("Obtain ground truth for missing names")
    print(obtain_ground_truth_name(100))