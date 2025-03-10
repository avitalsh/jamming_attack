import os


import argparse
import random
import numpy as np
import torch
import json
import copy
import pickle
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import gc

import jamming_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # general args
    parser.add_argument("--seed", type=int, default=0)

    # model args
    parser.add_argument("--llm_model", default='Llama-2-7b-chat-hf', choices=['Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Mistral-7B-Instruct-v0.2',  'vicuna-7b-v1.3', 'vicuna-13b-v1.3'])
    parser.add_argument("--emb_model", default='gtr-base', choices=['gtr-base', 'contriever'])
    parser.add_argument("--oracle_llm", default='gpt-4-1106-preview', choices=['gpt-4-1106-preview', 'gpt-4o-mini'])
    parser.add_argument('--max_response_len', type=int, default=128)
    parser.add_argument("--llm_batch_size", type=int, default=16, help="LLM batch size, to avoid OOM")

    # RAG args
    parser.add_argument("--dataset", default='nq', choices=['nq', 'msmarco'], help="Evaluated dataset")
    parser.add_argument("--num_queries", type=int, default=100, help="Num queries to evaluate")
    parser.add_argument("--k", type=int, default=5, help="Num retrieved documents")

    args = parser.parse_args()

    args.cache_dir = './cache'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # load clean results to avoid evaluating queries for which the unpoisoned system did not provide an answer
    clean_path = f"./results/rag_{args.emb_model}_x_{args.llm_model}/{args.dataset}/clean/seed_{args.seed}/"
    os.makedirs(clean_path, exist_ok=True)
    clean_path = os.path.join(clean_path, f"clean_results_k{args.k}_nq{args.num_queries}_oracle_llm_{args.oracle_llm}.pkl")
    if os.path.exists(clean_path):
        print("Clean results already evaluated")
        exit()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    print(f"load RAG embedding model: {args.emb_model}")
    emb_dict = jamming_utils.load_emb_model(args.emb_model, device)

    print(f"load dataset")
    dataloader, q_embs, q_embs_names, corpus = jamming_utils.load_queries(args.dataset, args.num_queries, emb_dict, args.seed)

    print("load BEIR results")
    beir_path = f'./data/beir_eval/{args.dataset}-{"dev" if args.dataset == "msmarco" else "test"}-{args.emb_model}-{emb_dict["score_func_name"]}.json'
    if not os.path.exists(beir_path):
        print(f"Please evaluate on BEIR first {beir_path}")
        exit(-1)
    with open(beir_path, 'r') as f:
        beir_results = json.load(f)

    print(f"-----START-----")
    res_path_temp = clean_path.replace('.pkl', '_temp.pkl')
    if os.path.exists(res_path_temp):
        print(f'Continue running from temp results')
        with open(res_path_temp, 'rb') as f:
            results = pickle.load(f)
    else:
        print("Run from scratch")
        results = {'last_idx': -1, 'queries': [], 'answers': [],
                   'clean_response_hist': [], #save the clean responses [#NUM_QUERIES],
                   'clean_response_answer_binary': [] #did the response answer the query [#NUM_QUERIES]
                   }
    print(f"load RAG llm model {args.llm_model}")
    llm_model, llm_params, conv_template = jamming_utils.load_llm_model(args, num_avail_gpus=torch.cuda.device_count())
    print(f"done loading models")

    train_iter = iter(dataloader)
    for q_idx in range(len(dataloader)):
        data = next(train_iter)
        cur_query_name = data[0]['qrels']
        cur_query_txt = data[0]['sent0']
        cur_answer_txt = data[0]['sent1']
        print(f"QUERY {q_idx}/{len(dataloader)} ({cur_query_name}) = {cur_query_txt}")

        if q_idx > results['last_idx']:  # didn't cache this yet
            cur_query_beir_res = copy.deepcopy(beir_results[cur_query_name])

            # get the original clean llm response for the retrieved results
            context_str_list, orig_contexts, context_names = jamming_utils.get_context_str(corpus, cur_query_beir_res,
                                                                                   [args.k], adv_docs=None)
            prompt_list = [jamming_utils.get_prompt(args, conv_template, context_str, cur_query_txt) for context_str in context_str_list]
            response = jamming_utils.get_llm_pred(llm_type=args.llm_model, llm_model=llm_model, prompt_list=prompt_list,
                                                  llm_params=llm_params, batch_size=args.llm_batch_size)[0]

            answer_bin = jamming_utils.check_if_answer(q=cur_query_txt, a=response, oracle_llm=args.oracle_llm)

            print(f"Response: {response}\nAnswered: {answer_bin}")

            results['last_idx'] = q_idx
            results['queries'].append(cur_query_txt)
            results['answers'].append(cur_answer_txt)
            results['clean_response_hist'].append(response)
            results['clean_response_answer_binary'].append(answer_bin)

            print("save temp..")
            with open(res_path_temp, 'wb') as f:
                pickle.dump(results, f)
    results['clean_response_answer_binary'] = np.array(results['clean_response_answer_binary'])
    with open(clean_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"save final results in {clean_path}")
    destroy_model_parallel()
    # del llm_model.llm_engine.model_executor
    del llm_model
    gc.collect()
    torch.cuda.empty_cache()
    import ray

    ray.shutdown()
    print("done cleaning memory and processes")
    os.remove(res_path_temp)
    print(f"remove temp file from {res_path_temp}")




