import fasttext
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/workspace/datasets/fasttext/norm_model.bin")
parser.add_argument("--threshold", default=0.75)
parser.add_argument("--output_path", default="/workspace/datasets/fasttext/synonyms.csv")
parser.add_argument("--input_path", default="/workspace/datasets/fasttext/top_words.txt")

args = parser.parse_args()

if __name__ == "__main__":
    model = fasttext.load_model(args.model)
    result = []
    with open(args.input_path) as f:
        for line in f.readlines():
            line = line.strip()
            pred = model.get_nearest_neighbors(line)
            for score, word in pred:
                if score > args.threshold:
                    line += "," + word
            result.append(line)
    with open(args.output_path, "w") as f:
        for r in result:
            f.write("{}\n".format(r))
            