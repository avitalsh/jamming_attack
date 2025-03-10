#!/bin/bash


RAG_LLM=Llama-2-7b-chat-hf
RAG_EMB=gtr-base
SCORE_FUNC=cos_sim
DS=nq
SPLIT=test
NUM_Q=100
O_EMB=text-embedding-3-small
NUM_T=50
BS=32
RES_TAR=t1

python eval_beir.py --model_code $RAG_EMB --score_function $SCORE_FUNC --dataset $DS --split $SPLIT

python get_clean_responses.py --llm_model $RAG_LLM --emb_model $RAG_EMB --dataset $DS --k 5 --num_queries $NUM_Q

python attack.py --llm_model $RAG_LLM --emb_model $RAG_EMB --dataset $DS --k 5 --num_queries $NUM_Q --oracle_emb_model $O_EMB --num_tokens $NUM_T --num_iterations 1000 --es_iterations 100 --batch_size $BS --doc_init "mask" --response_target $RES_TAR