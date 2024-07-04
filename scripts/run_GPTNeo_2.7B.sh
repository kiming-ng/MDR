#!/bin/bash

method=mdr-test # should start with 'mdr'

PROJ_DIR=path_to_MDR;
cd ${PROJ_DIR};

export CUDA_VISIBLE_DEVICES=0;

RETRIEVER=path_to_retriever;
# select demon_pool & LLM path & set LLM name
DEMONSTRATION_POOL=./demonstration_pools/demonstration_pool_GPTNeo.json;
LLM=EleutherAI/gpt-neo-2.7B; # GPT-Neo-2.7B
LLM_NAME=GPTNeo;

CACHE_DIR=path_to_cache_dir; # for model and dataset downloading
ENCODED_POOL_DIR=$PROJ_DIR/uprise/encode_pool_uprise;

# Unseed Tasks
# TASKS=('java' 'python' 'samsum' 'squad_v2' 'trivia_qa' 'wmt14' 'wmt16' );
# TASKS=('cosmos_qa' 'wnli' 'anli' 'amazon' 'mr' 'rotten_tomatoes' 'wic' 'wsc273');

TASKS=('cosmos_qa');

for TASK in "${TASKS[@]}"
do

	k_similarity=20; # K'
	score_layer=32;
	num_deomns=(3); # K
	Cs=(0.9); # C

	EXP_DIR=$PROJ_DIR/experiments/$method/$LLM_NAME/$TASK/layer_$score_layer/total_doc_$k_similarity;

	# retrieve prompts for each task example
	# the retrieved prompts will be in '$EXP_DIR/${TASK}_prompts.json'
	python DPR/dense_retriever.py \
		model_file=${RETRIEVER} \
		qa_dataset=qa_uprise \
		ctx_datatsets=[dpr_uprise] \
		encoded_ctx_files=[$ENCODED_POOL_DIR/dpr_enc_index_*] \
		out_file=$EXP_DIR/${TASK}_prompts_doc$k_similarity.json \
		datasets.qa_uprise.task_name=${TASK} \
		datasets.qa_uprise.cache_dir=${CACHE_DIR} \
		n_docs=$k_similarity \
		method=$method \
		score.layer=$score_layer \
		score.batch_size=1 \
		score.new_prompt_pool_path=${DEMONSTRATION_POOL} \
		score.dataset_reader.model_name=$LLM \
		ctx_sources.dpr_uprise.prompt_pool_path=${DEMONSTRATION_POOL} \
		ctx_sources.dpr_uprise.prompt_setup_type=qa \
		encoder.cache_dir=${CACHE_DIR} \
		hydra.run.dir=$EXP_DIR/retrieve_log \
		seed=$seed;

	# rerank & inference
	for C in "${Cs[@]}"
	do
		lamda_DIR=lamda_$C;
		mkdir -p $EXP_DIR/$lamda_DIR/;
		python src/utils/get_mix_prompt.py --src_prompt_file $EXP_DIR/${TASK}_prompts_doc$k_similarity.json --tgt_prompt_file $EXP_DIR/$lamda_DIR/${TASK}_prompts_doc$k_similarity\_lamda$C.json --lamda $C;
		
		# inference with different `num_prompt`
		for n_d in "${num_deomns[@]}"
		do
			python inference.py \
				prompt_file=$EXP_DIR/$lamda_DIR/${TASK}_prompts_doc$k_similarity\_lamda$C.json \
				task_name=${TASK} \
				output_file=$EXP_DIR/$lamda_DIR/${TASK}_pred_np_$n_d.json \
				res_file=$EXP_DIR/$lamda_DIR/${TASK}_evaluation_res.txt \
				model_name=${LLM} \
				cache_dir=${CACHE_DIR} \
				num_prompts=$n_d \
				batch_size=1 \
				hydra.run.dir=$EXP_DIR/inference_log;
		done;
	done;
done;