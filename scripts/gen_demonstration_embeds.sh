#!/bin/bash

PROJ_DIR=path_to_MDR

RETRIEVER=path_to_retriever # path to the downloaded retriever checkpoint
PROMPT_POOL=path_to_demonstration_pool # path to the downloaded demonstration pool
CACHE_DIR=path_to_cache_dir # directory path for caching the LLM checkpoints, task datasets, etc.

cd $PROJ_DIR;

python DPR/generate_dense_embeddings.py \
	 model_file=${RETRIEVER} \
	 ctx_src=dpr_uprise shard_id=0 num_shards=1 \
	 out_file=$PROJ_DIR/uprise/encode_pool_uprise/dpr_enc_index \
	 ctx_sources.dpr_uprise.prompt_pool_path=${PROMPT_POOL} \
	 ctx_sources.dpr_uprise.prompt_setup_type=qa \
	 encoder.cache_dir=${CACHE_DIR} \
	 hydra.run.dir=$PROJ_DIR/uprise/encode_pool_uprise;