"""
Collect document‑frequency for every BPE token and save smooth‑IDF
table to disk as a numpy array (len == tokenizer.vocab_size).

Run:
    python build_tfidf.py --root /home/cis/cc3m-ironic \
                          --tok answerdotai/ModernBERT-base \
                          --out  /home/cis/cc3m-ironic/idf.npy
"""
import argparse, math, json, numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

def main(root: Path, tok_name: str, out: Path):
    tk = AutoTokenizer.from_pretrained(tok_name)
    df = np.zeros(tk.vocab_size, dtype=np.int32)
    n_docs = 0

    for txt_file in tqdm(root.rglob("*.txt")):
        n_docs += 1
        ids = tk(txt_file.read_text(), add_special_tokens=False)["input_ids"]
        for tid in set(ids):                     # document‑frequency
            df[tid] += 1

    # smooth‑idf: log((N + 1)/(df + 1)) + 1   (always ≥ 1)
    idf = np.log((n_docs + 1) / (df + 1)) + 1.0
    np.save(out, idf.astype(np.float32))
    print(f"Saved IDF for {idf.size} tokens to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default="/home/cis/cc3m-ironic")
    p.add_argument("--tok_name",  type=str, default="distilbert/distilbert-base-uncased")
    p.add_argument("--out",  type=Path, default="/home/cis/cc3m-ironic/idf.npy")
    main(**vars(p.parse_args())) 