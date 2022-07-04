import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="/workspace/datasets/fasttext/label_products.txt",  help="input path")
parser.add_argument("--min", default=500,  help="min count")
parser.add_argument("--output_path", default="/workspace/datasets/fasttext/pruned_label_products.txt",  help="min count")

args = parser.parse_args()

def read_data(path):
    with open(path) as f:
        data = f.readlines()
    split_data = []
    for line in data:
        s = line.rstrip().split(" ", 1)
        split_data.append(s)
    df = pd.DataFrame(split_data, columns=["category", "text"])
    counts = df['category'].value_counts()
    filtered = counts[lambda x : x > args.min].index
    result = df[df.category.isin(filtered)]
    with open(args.output_path, "w") as f:
        for idx, row in result.iterrows():
            f.write("{} {}\n".format(row['category'], row['text']))


if __name__ == "__main__":
    read_data(args.path)
    


