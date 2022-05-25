import json
import os
from tqdm.auto import tqdm

def read_neighbor_model(filename):
    neighbor_map = {}
    with open(filename, "r") as f:
        neighbor_map = json.load(f)
    return neighbor_map

def complement_vocab(filename, neighbor_model):
    target_vocab = set()
    print("reading current vocab")
    with open(filename, "r") as f:
        for line in tqdm(f):
            tok = line.rstrip()
            if tok[:3] != "en_":
                continue
            target_tok = neighbor_model.get(tok,None)
            if target_tok: target_vocab.update([tok[0] for tok in target_tok[:3]])
    
    print("adding vocab to existing")
    with open(filename, "a") as f:
        for tok in tqdm(target_vocab):
            f.write(f"{tok}\n")
    print(f"done for {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('--neighbor_model', default='data/numberbatch/en_ms_neighbors_model.json')
    parser.add_argument('--vocab_dir', default="data/vocabulary")
    args = parser.parse_args()

    neighbor_map = read_neighbor_model(args.neighbor_model)

    complement_vocab(os.path.join(args.vocab_dir, "encoder_token_ids.txt"), neighbor_map)
