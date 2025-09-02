cd ..
split="7_0.2" # change it for different splits
base_model="EleutherAI/pythia-12b-deduped" # change it for different models
for data_source in arxiv_ngram_$split github_ngram_$split hackernews_ngram_$split dm_mathematics_ngram_$split  pile_cc_ngram_$split pubmed_central_ngram_$split
do
    CUDA_VISIBLE_DEVICES=0  python run_baselines.py specific_source=$data_source experiment_name=baselines_$data_source base_model=$base_model
    CUDA_VISIBLE_DEVICES=0  python run_ref_baselines.py specific_source=$data_source base_model=$base_model
    CUDA_VISIBLE_DEVICES=0  python run_ours_construct_mia_data.py specific_source=$data_source base_model=$base_model
    CUDA_VISIBLE_DEVICES=0  python run_ours_ours_train_lr.py specific_source=$data_source base_model=$base_model
    CUDA_VISIBLE_DEVICES=0  python run_ours_ours_different_agg.py specific_source=$data_source base_model=$base_model
    CUDA_VISIBLE_DEVICES=0  python run_ours_ours_get_roc.py specific_source=$data_source base_model=$base_model
done

