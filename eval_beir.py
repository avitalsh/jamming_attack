import logging
import pathlib, os
import json
import torch
import sys
import transformers

sys.path.append('corpus_poisoning/src')
sys.path.append('corpus_poisoning/src/beir')
sys.path.append('corpus_poisoning/src/contriever')
sys.path.append('corpus_poisoning/src/contriever/src')
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from contriever import Contriever
from beir_utils import DenseEncoderModel

import numpy as np
from typing import List, Dict

import argparse

import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--model_code', type=str, default="contriever")
parser.add_argument('--score_function', type=str, default='dot',choices=['dot', 'cos_sim'])

parser.add_argument('--dataset', type=str, default="fiqa", help='BEIR dataset to evaluate')
parser.add_argument('--split', type=str, default='test')

parser.add_argument("--per_gpu_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
parser.add_argument('--max_length', type=int, default=128)

args = parser.parse_args()

from utils import model_code_to_cmodel_name, model_code_to_qmodel_name

def mean_pool(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs

class GTRembeddings:
    #this is for supporting gtr-base which is not included by default in the corpus poisoning code
    def __init__(self, **kwargs):
        model_kwargs = {
            "low_cpu_mem_usage": True,  # Not compatible with DeepSpeed
            "output_hidden_states": False,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedder = transformers.AutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-base", **model_kwargs
        ).encoder.to(self.device)
        self.embedder.eval()
        self.embedder_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
        self.max_seq_length = 128

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        for k in range(nbatch):
            print(f"queries, batch {k}/{nbatch}")
            start_idx = k * batch_size
            end_idx = min((k + 1) * batch_size, len(queries))
            batch = queries[start_idx:end_idx]
            with torch.no_grad():
                sent_tokens = self.embedder_tokenizer(batch,
                                                      padding="max_length",
                                                      truncation=True,
                                                      max_length=self.max_seq_length,
                                                      return_tensors="pt").to(self.device)
                hidden_state = self.embedder(input_ids=sent_tokens['input_ids'],
                                                    attention_mask=sent_tokens['attention_mask']).last_hidden_state
                emb = mean_pool(hidden_state, sent_tokens['attention_mask']).cpu().numpy()
            allemb.append(emb)
        allemb = np.concatenate(allemb, axis=0)
        print(len(queries), allemb.shape)
        return allemb

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]

        allemb = []
        nbatch = (len(corpus) - 1) // batch_size + 1
        for k in range(nbatch):
            print(f"corpus, batch {k}/{nbatch}")
            start_idx = k * batch_size
            end_idx = min((k + 1) * batch_size, len(corpus))
            batch = corpus[start_idx:end_idx]
            with torch.no_grad():
                sent_tokens = self.embedder_tokenizer(batch,
                                                      padding="max_length",
                                                      truncation=True,
                                                      max_length=self.max_seq_length,
                                                      return_tensors="pt").to(self.device)
                hidden_state = self.embedder(input_ids=sent_tokens['input_ids'],
                                             attention_mask=sent_tokens['attention_mask']).last_hidden_state
                emb = mean_pool(hidden_state, sent_tokens['attention_mask']).cpu().numpy()
            allemb.append(emb)
        allemb = np.concatenate(allemb, axis=0)
        print(len(corpus), allemb.shape)
        return allemb


def compress(results):
    for y in results:
        k_old = len(results[y])
        break
    sub_results = {}
    for query_id in results:
        sims = list(results[query_id].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        sub_results[query_id] = {}
        for c_id, s in sims[:2000]:
            sub_results[query_id][c_id] = s
    for y in sub_results:
        k_new = len(sub_results[y])
        break
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

logging.info(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('./data/beir_eval/', exist_ok=True)
args.result_output = f'./data/beir_eval/{args.dataset}-{args.split}-{args.model_code}-{args.score_function}.json'

if os.path.exists(args.result_output):
    print(f"BEIR results already evaluated and saved at: {args.result_output}")
    exit()

#### Download and load dataset
dataset = args.dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = "./corpus_poisoning/datasets"
os.makedirs(out_dir, exist_ok=True)
data_path = os.path.join(out_dir, dataset)
if not os.path.exists(data_path):
    data_path = util.download_and_unzip(url, out_dir)
logging.info(data_path)

corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)

logging.info("Loading model...")
if 'contriever' in args.model_code:
    encoder = Contriever.from_pretrained(model_code_to_cmodel_name[args.model_code]).cuda()
    tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[args.model_code])
    model = DRES(DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer), batch_size=args.per_gpu_batch_size)
elif args.model_code == 'gtr-base':
    model = DRES(GTRembeddings(), batch_size=args.per_gpu_batch_size)
else:
    raise NotImplementedError

logging.info(f"model: {model.model}")

retriever = EvaluateRetrieval(model, score_function=args.score_function, k_values=[1,3,5,10,20,100,1000]) # "cos_sim"  or "dot" for dot-product
results = retriever.retrieve(corpus, queries)

logging.info("Printing results to %s"%(args.result_output))
sub_results = compress(results)

with open(args.result_output, 'w') as f:
    json.dump(sub_results, f)




