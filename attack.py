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

from transformers import default_data_collator


import jamming_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # general args
    parser.add_argument("--seed", type=int, default=0)

    # model args
    parser.add_argument("--llm_model", default='Llama-2-7b-chat-hf', choices=['Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Mistral-7B-Instruct-v0.2',  'vicuna-7b-v1.3', 'vicuna-13b-v1.3'])
    parser.add_argument("--emb_model", default='gtr-base', choices=['gtr-base', 'contriever'])
    parser.add_argument("--oracle_emb_model", default='text-embedding-3-small', choices=['text-embedding-3-small', 'gtr-base', 'contriever'])
    parser.add_argument("--oracle_llm", default='gpt-4-1106-preview', choices=['gpt-4-1106-preview', 'gpt-4o-mini'])
    parser.add_argument('--max_response_len', type=int, default=128)
    parser.add_argument("--llm_batch_size", type=int, default=16, help="LLM batch size, to avoid OOM")

    # RAG args
    parser.add_argument("--dataset", default='nq', choices=['nq', 'msmarco'], help="Evaluated dataset")
    parser.add_argument("--num_queries", type=int, default=100, help="Num queries to evaluate")
    parser.add_argument("--k", type=int, default=5, help="Num retrieved documents")

    #experiment args
    parser.add_argument("--num_tokens", type=int, default=50, help="Number of tokens in the blocker document")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of optimization iterations")
    parser.add_argument("--es_iterations", type=int, default=100, help="Number of iterations for early stop")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of candidates in each iteration")
    parser.add_argument("--doc_init", default='mask', choices=['random', 'mask'], help="Initialization method for blocker document")
    parser.add_argument("--response_target", default='t1', choices=['t1', 't2', 't3'], help="The target response")

    args = parser.parse_args()

    args.cache_dir = './cache'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    results_dir = f'./results/rag_{args.emb_model}_x_{args.llm_model}/{args.dataset}/oracle_{args.oracle_emb_model}/seed_{args.seed}/'
    os.makedirs(results_dir, exist_ok=True)



    # load clean results to avoid evaluating queries for which the unpoisoned system did not provide an answer
    clean_path = f"./results/rag_{args.emb_model}_x_{args.llm_model}/{args.dataset}/clean/seed_{args.seed}/clean_results_k{args.k}_nq{args.num_queries}_oracle_llm_{args.oracle_llm}.pkl"
    if not os.path.exists(clean_path):
        print("Please evaluate clean results first")
        exit(-1)
    else:
        print(f"Clean results loaded from {clean_path}")
        with open(clean_path, 'rb') as f:
            clean_results = pickle.load(f)

    res_path = os.path.join(results_dir, f'results_k{args.k}_nt{args.num_tokens}_i{args.num_iterations}_es{args.es_iterations}_bs{args.batch_size}_init_{args.doc_init}_res_tar_{args.response_target}_nq{args.num_queries}_oracle_llm_{args.oracle_llm}.pkl')
    if os.path.exists(res_path):
        with open(res_path, 'rb') as f:
            results = pickle.load(f)
    else:

        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        print(f"load RAG embedding model: {args.emb_model}")
        emb_dict = jamming_utils.load_emb_model(args.emb_model, device)
        score_func_rag = emb_dict['score_func']

        print(f"load oracle embedding model: {args.oracle_emb_model}")
        oracle_emb_dict = jamming_utils.load_emb_model(args.oracle_emb_model, device)
        score_func_oracle = oracle_emb_dict['score_func']

        print(f"load token vocabulary")
        token_vocab, mask_token_id = jamming_utils.get_vocab_and_mask(args.oracle_emb_model, oracle_emb_dict)
        # get token probability for sampling
        token_probs_path = f"./data/tokenizers/{args.oracle_emb_model}_wikitext_stats.pkl"
        if not os.path.exists(token_probs_path):
            os.makedirs("./data/tokenizers/", exist_ok=True)
            print(f"preprocess token stats using wikitext")
            token_counts, token_probs = jamming_utils.compute_token_stats(token_vocab, oracle_emb_dict)
            with open(token_probs_path, 'wb') as f:
                pickle.dump({'token_probs': token_probs, 'token_counts': token_counts}, f)
        with open(token_probs_path, 'rb') as f:
            token_stats = pickle.load(f)
        token_probs = token_stats['token_probs']
        token_counts = token_stats['token_counts']
        # we filter out the 100 most popular tokens
        ids = np.argsort(token_counts)[::-1][:100]
        token_vocab = np.delete(token_vocab, ids)
        token_counts = np.delete(token_counts, ids)
        token_probs = token_counts / np.sum(token_counts)


        print(f"load dataset")
        dataloader, q_embs, q_embs_names, corpus = jamming_utils.load_queries(args.dataset, args.num_queries, emb_dict,
                                                                              args.seed)

        print("load BEIR results")
        beir_path = f'./data/beir_eval/{args.dataset}-{"dev" if args.dataset == "msmarco" else "test"}-{args.emb_model}-{emb_dict["score_func_name"]}.json'
        if not os.path.exists(beir_path):
            print(f"Please evaluate on BEIR first {beir_path}")
            exit(-1)
        with open(beir_path, 'r') as f:
            beir_results = json.load(f)

        res_target, res_target_emb_oracle = jamming_utils.get_res_target(args.response_target, oracle_emb_dict)
        print(f"-----START-----")
        res_path_temp = res_path.replace('.pkl', '_temp.pkl')
        if os.path.exists(res_path_temp):
            print(f'Continue running from temp results')
            with open(res_path_temp, 'rb') as f:
                results = pickle.load(f)
        else:
            print("Run from scratch")
            results = {'last_idx': -1, 'queries': [], 'answers': [],
                       'doc_hist': [], # history of blocker documents during optimization [#NUM_QUERIES x max(#ITERATIONS)]
                       'dist_doc2query': [], # history of distance (based on RAG emb) between doc to query, should be close enough to be retrieved [#NUM_QUERIES x #ITERATIONS]
                       'dist_res2res_target': [], # histroy of the distance between the poisoned RAG response and the target response, same as the loss [#NUM_QUERIES x #ITERATIONS]
                       'dist_res2clean_res': [], # histroy of the distance between the poisoned RAG response and the clean response [#NUM_QUERIES x #ITERATIONS]
                       'response_hist': [],  # save the responses [#NUM_QUERIES x max(#ITERATIONS)]
                       'final_response_answer_binary': [],  # did the final response answer the query [#NUM_QUERIES]
                       'loss_hist': [], # loss history [#NUM_QUERIES x #ITERATIONS]
                       'early_stop': [], # the iteration in which we stopped, if < args.iterations it means there was an early stop
                       'sampled_token_hist': [], # history of all sampled tokens in optimization. [#NUM_QUERIES x #ITERATIONS x #BATCH_SIZE]
                       'rep_loc_hist': [], # history of sampled index to replace in optimization [#NUM_QUERIES x #ITERATIONS]
                       'idx_chosen_sampled_token_hist': [], # history of which token was selected in optimization (token index), if non then -1 [#NUM_QUERIES x #ITERATIONS]
                       'val_chosen_sampled_token_hist': [], # history of which token was selected in optimization (decoded token), if non then -1 [#NUM_QUERIES x #ITERATIONS]
                       'ret_hist': [] # if the doc was retrieved [#NUM_QUERIES x #ITERATIONS]
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
                data = default_data_collator(data)
                cur_query_emb = q_embs[np.where(q_embs_names == cur_query_name)[0][0]].reshape(1, -1)

                if clean_results["clean_response_answer_binary"][q_idx] == 0:
                    print(f"SKIPPING - was not answered by the unpoisond RAG system")

                    results['last_idx'] = q_idx
                    results['queries'].append(cur_query_txt)
                    results['answers'].append(cur_answer_txt)
                    results['doc_hist'].append([""])
                    results['response_hist'].append([""])
                    results['final_response_answer_binary'].append(-10)
                    results['early_stop'].append(0)

                    results['sampled_token_hist'].append([-1])
                    results['rep_loc_hist'].append([-1])
                    results['idx_chosen_sampled_token_hist'].append([-1])
                    results['val_chosen_sampled_token_hist'].append([-1])

                    #we numpy these things later so they need to have the same shape
                    results['dist_doc2query'].append([0] * (args.num_iterations + 1))
                    results['dist_res2res_target'].append([0] * (args.num_iterations + 1))
                    results['dist_res2clean_res'].append([0] * (args.num_iterations + 1))
                    results['loss_hist'].append([0] * (args.num_iterations + 1))
                    results[f'ret_hist'].append([-1] * (args.num_iterations + 1))

                    continue

                clean_response_txt = clean_results[f'clean_response_hist'][q_idx]
                clean_response_emb_oracle = jamming_utils.get_embedding(text=clean_response_txt, emb_dict=oracle_emb_dict)

                cur_adv_doc_list, cur_adv_doc_txt = jamming_utils.init_adv_doc(args, mask_token_id, token_vocab,
                                                                               tokenizer=oracle_emb_dict['tokenizer'])
                #for ensuring retrieval, we prepend the query itself
                cur_adv_doc_txt = cur_query_txt + ". " + cur_adv_doc_txt
                cur_adv_doc_emb = jamming_utils.get_embedding(cur_adv_doc_txt, emb_dict)
                cur_adv_doc_emb_oracle = jamming_utils.get_embedding(cur_adv_doc_txt, oracle_emb_dict)
                print(f"Cur blocker doc = {cur_adv_doc_txt} ({cur_adv_doc_emb_oracle.shape}, {cur_adv_doc_emb.shape})")

                print(f"Clean response: {clean_response_txt}")

                print(f"collect initial metrics")
                doc_hist = [cur_adv_doc_txt]

                dist_doc2query = [np.mean(score_func_rag(cur_query_emb, cur_adv_doc_emb))]

                cur_context_str_list, cur_contexts, cur_contexts_names = jamming_utils.get_context_str(corpus, cur_query_beir_res, [args.k],
                                                                                                       adv_docs={'adv': {'dist': dist_doc2query[-1], 'txt': cur_adv_doc_txt}})
                adv_ret_flag = 'adv' in cur_contexts_names
                if adv_ret_flag:
                    #was retrieved so run llm
                    prompt_list = [jamming_utils.get_prompt(args, conv_template, context_str, cur_query_txt) for context_str in
                                            cur_context_str_list] # this is a table because we can technically evaluated multiple k vals at once
                    response = jamming_utils.get_llm_pred(llm_type=args.llm_model, llm_model=llm_model, prompt_list=prompt_list,
                                                  llm_params=llm_params, batch_size=args.llm_batch_size)[0]

                    res_emb_oracle = jamming_utils.get_embedding(text=response, emb_dict=oracle_emb_dict)
                    ret_hist = [1]
                    response_hist = [response]
                    dist_res2res_target = [np.mean(np.abs(score_func_oracle(res_emb_oracle, res_target_emb_oracle)))]
                    dist_res2clean_res = [np.mean(np.abs(score_func_oracle(res_emb_oracle, clean_response_emb_oracle)))]
                else:
                    respone = clean_response_txt
                    res_emb_oracle = clean_response_emb_oracle
                    ret_hist = [0]
                    response_hist = [clean_response_txt]
                    dist_res2res_target = [np.mean(np.abs(score_func_oracle(clean_response_emb_oracle, res_target_emb_oracle)))]
                    dist_res2clean_res = [0.]

                cur_loss = dist_res2res_target[-1]
                loss_hist = [cur_loss]

                sampled_token_hist, rep_loc_hist, idx_chosen_sampled_token_hist, val_chosen_sampled_token_hist = [], [], [], []

                print("-" * 100)
                print(f"start optimizing")
                es_count = 0
                for iter_idx in range(args.num_iterations):
                    ####################################################################################################
                    print("=" * 100)
                    print(f"Q{q_idx}: {iter_idx}/{args.num_iterations}, ES={es_count}, Retrieved={ret_hist[-1]}\n"
                          f"Loss={loss_hist[-1]}, D_doc2query={dist_doc2query[-1]}, D_res2res_tar={dist_res2res_target[-1]}, D_res2clean_res={dist_res2clean_res[-1]}\n"
                          f"DOC: {doc_hist[-1]}\n\n"
                          f"RES: {response_hist[-1]}\n")
                    if iter_idx >= 1:
                        print(f"Rep_loc: {rep_loc_hist[-1]}, idx_token: {idx_chosen_sampled_token_hist[-1]}, val_token: {val_chosen_sampled_token_hist[-1]}")
                    print("=" * 100)
                    ####################################################################################################

                    # create replacement candidates
                    cands = np.sort(rng.choice(token_vocab, size=args.batch_size, replace=False, p=token_probs))
                    sampled_token_hist.append(cands)

                    # sample token to be replaced
                    loc = rng.choice(args.num_tokens)
                    rep_loc_hist.append(loc)

                    # replace
                    cand_docs_list = np.array([cur_adv_doc_list] * args.batch_size)
                    cand_docs_list[:, loc] = cands

                    # convert tokens to text
                    cand_docs_txt = oracle_emb_dict['tokenizer'].decode_batch(cand_docs_list)
                    # add query for retrieval
                    for c_idx in range(len(cand_docs_txt)):
                        cand_docs_txt[c_idx] = cur_query_txt + ". " + cand_docs_txt[c_idx]
                    #compute embeddings for measuring distacnes
                    batch_embs = jamming_utils.get_embedding(text=cand_docs_txt, emb_dict=emb_dict)
                    print(batch_embs.shape)

                    batch_dist_doc2query = np.mean(score_func_rag(batch_embs, cur_query_emb), axis=1)

                    # get llm responses for each candidate
                    prompt_list, found_list = [], []
                    for c_idx in range(args.batch_size):
                        adv_doc_dict = {'adv': {'dist': batch_dist_doc2query[c_idx], 'txt': cand_docs_txt[c_idx]}}
                        cur_context_str_list, cur_contexts, cur_contexts_names = jamming_utils.get_context_str(corpus,
                                                                                                       cur_query_beir_res,
                                                                                                       [args.k],
                                                                                                       adv_docs=adv_doc_dict)
                        adv_ret_flag = 'adv' in cur_contexts_names
                        found_list.append(adv_ret_flag)
                        if adv_ret_flag:
                            # only run llm when the doc was retrieved
                            prompt_list.append(jamming_utils.get_prompt(args, conv_template, cur_context_str_list[0], cur_query_txt))
                    response = jamming_utils.get_llm_pred(llm_type=args.llm_model, llm_model=llm_model, prompt_list=prompt_list,
                                                  llm_params=llm_params, batch_size=args.llm_batch_size)

                    if len(response) > 0:
                        new_res_emb_oracle = jamming_utils.get_embedding(text=response, emb_dict=oracle_emb_dict)
                    # aggregate all responses and embeddings, take the clean ones when the doc wasn't retrieved
                    r_idx = 0
                    res_emb_oracle, cand_responses = [], []
                    for f in found_list:
                        if f:
                            res_emb_oracle.append(new_res_emb_oracle[r_idx].reshape(1, -1))
                            cand_responses.append(response[r_idx])
                            r_idx += 1
                        else:
                            res_emb_oracle.append(clean_response_emb_oracle)
                            cand_responses.append(clean_response_txt)
                    res_emb_oracle = np.concatenate(res_emb_oracle)

                    batch_dist_res2res_target = np.mean(np.abs(score_func_oracle(res_emb_oracle, res_target_emb_oracle)), axis=1)
                    loss = - batch_dist_res2res_target
                    best_cand_idx = np.argmin(loss)

                    if loss[best_cand_idx] < cur_loss:
                        print(f"replace\n{cur_adv_doc_txt}\n-WITH-\n{cand_docs_txt[best_cand_idx]}")
                        cur_adv_doc_txt = cand_docs_txt[best_cand_idx]
                        idx_chosen_sampled_token_hist.append(best_cand_idx)
                        val_chosen_sampled_token_hist.append(cands[best_cand_idx])

                        doc_hist.append(cur_adv_doc_txt)
                        dist_doc2query.append(batch_dist_doc2query[best_cand_idx])
                        ret_hist.append(1 if found_list[best_cand_idx] else 0)
                        response_hist.append(cand_responses[best_cand_idx])
                        dist_res2res_target.append(batch_dist_res2res_target[best_cand_idx])
                        dist_res2clean_res.append(np.mean(np.abs(score_func_oracle(res_emb_oracle[best_cand_idx].reshape(1, -1), clean_response_emb_oracle))))

                        cur_loss = loss[best_cand_idx]
                    else:
                        # no change in this iter
                        idx_chosen_sampled_token_hist.append(-1)
                        val_chosen_sampled_token_hist.append(-1)
                        es_count += 1
                        # no change in this iter, just repeat the prev values
                        dist_doc2query.append(dist_doc2query[-1])
                        ret_hist.append(ret_hist[-1])
                        dist_res2res_target.append(dist_res2res_target[-1])
                        dist_res2clean_res.append(dist_res2clean_res[-1])

                    loss_hist.append(cur_loss)

                    if es_count >= args.es_iterations or jamming_utils.check_if_i_dont_know(response_hist[-1]):
                        print(f"terminating!")
                        print(f"es_count = {es_count}. response = {response_hist[-1]}")

                        # again, for shape consistency, fill in the remaining values
                        for rem_iter in range(iter_idx + 1, args.num_iterations):
                            dist_doc2query.append(dist_doc2query[-1])
                            dist_res2res_target.append(dist_res2res_target[-1])
                            dist_res2clean_res.append(dist_res2clean_res[-1])
                            loss_hist.append(cur_loss)
                            ret_hist.append(ret_hist[-1])

                        break

                if ret_hist[-1] == 1:
                    answer_bin = jamming_utils.check_if_answer(q=cur_query_txt, a=response_hist[-1], oracle_llm=args.oracle_llm)
                else:
                    answer_bin = -10
                print(f"Did final response answer: {answer_bin}")

                #done optimizing this doc
                results['last_idx'] = q_idx
                results['queries'].append(cur_query_txt)
                results['answers'].append(cur_answer_txt)
                results['doc_hist'].append(doc_hist)
                results['response_hist'].append(response_hist)
                results['early_stop'].append(iter_idx)

                results['sampled_token_hist'].append(sampled_token_hist)
                results['rep_loc_hist'].append(rep_loc_hist)
                results['idx_chosen_sampled_token_hist'].append(idx_chosen_sampled_token_hist)
                results['val_chosen_sampled_token_hist'].append(val_chosen_sampled_token_hist)

                results['dist_doc2query'].append(dist_doc2query)
                results['dist_res2res_target'].append(dist_res2res_target)
                results['dist_res2clean_res'].append(dist_res2clean_res)
                results['loss_hist'].append(loss_hist)
                results[f'ret_hist'].append(ret_hist)

                results['final_response_answer_binary'].append(answer_bin)

                print("save temp")
                with open(res_path_temp, 'wb') as f:
                    pickle.dump(results, f)

        print(f"done evaluating all queries")

        results['early_stop'] = np.array(results['early_stop'])
        results['dist_doc2query'] = np.array(results['dist_doc2query'])
        results['dist_res2res_target'] = np.array(results['dist_res2res_target'])
        results['loss_hist'] = np.array(results['loss_hist'])
        results['ret_hist'] = np.array(results['ret_hist'])
        results['final_response_answer_binary'] = np.array(results['final_response_answer_binary'])

        with open(res_path, 'wb') as f:
            pickle.dump(results, f)
        print("save final results")
        os.remove(res_path_temp)
        print("remove temp file")

    inds_rel_queries = np.where(clean_results["clean_response_answer_binary"] == 1)[0]
    print(f"We discard {args.num_queries-len(inds_rel_queries)}/{args.num_queries} for which the unpoisoned response did not provide an answer")

    inds_jammed = np.where(results['final_response_answer_binary'][inds_rel_queries] == 0)[0]
    print(f"{len(inds_jammed)}/{len(inds_rel_queries)} ({len(inds_jammed)/len(inds_rel_queries)}) were jammed.")









