# Running all evaluation includes
# COCO captioning, COCO self-retrieval
# nocap captioning, Flickr30k self-retrieval
# e.g., for our controllable model: bash scripts/eval_all cc3m_noc 1
# e.g., for vanilla or vanilla (filtering) baselines: bash scripts/eval_all cc3m_filtering 0 

expr_name=$1  # e.g., "cc3m_noc"
controllable=$2

PYTHONPATH=./ python3 scripts/make_link_to_best_model.py --expr-name ${expr_name}
bash scripts/eval_coco_cc.sh ${expr_name}
bash scripts/eval_nocap_cc.sh ${expr_name}
bash scripts/eval_flickr_cc.sh ${expr_name}

if [ ${controllable} -eq 1 ]
then
	roop=$(seq 0 7)
	for i in ${roop}
	do
		bash scripts/eval_retrieval.sh "results/$1/model/prediction_greedy_coco_bin${i}.json"
	done

	for i in ${roop}
	do
		bash scripts/eval_retrieval.sh "results/$1/model/prediction_greedy_flickr30k_bin${i}.json"
	done

else
	bash scripts/eval_retrieval.sh "results/$1/model/prediction_greedy_coco.json"
	bash scripts/eval_retrieval.sh "results/$1/model/prediction_greedy_flickr30k.json"
fi
