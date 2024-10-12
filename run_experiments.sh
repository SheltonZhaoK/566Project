#!/bin/bash
###
# CUB importance reduction of concepts w/ 112 residuals---------------------------------------------------------
###

# python3 experiments.py cub Joint -log_dir CUB/output/manuscript/112r_ig/detached_c_r/ -reduce ig -ray -ckpt 1 -should_detach -residue 112 -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
# python3 CUB/inference.py -ray -mi -crossC -reduce ig -residue 112 -model_dirs CUB/output/manuscript/112r_ig/detached_c_r/ CUB/output/manuscript/112r_ig/detached_c_r/ CUB/output/manuscript/112r_ig/detached_c_r/ -log_dir CUB/output/manuscript/112r_ig/detached_c_r/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
# python3 CUB/negTTI.py -ray -reduce ig -residue 112 -model_dirs CUB/output/manuscript/112r_ig/detached_c_r/ CUB/output/manuscript/112r_ig/detached_c_r/ CUB/output/manuscript/112r_ig/detached_c_r/ -log_dir CUB/output/manuscript/112r_ig/detached_c_r/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -log_dir CUB/output/manuscript/112r_ig/iterNorm/ -reduce ig -ray -ckpt 1 -should_detach -residue 112 -disentangle IterNorm -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
# python3 CUB/inference.py -ray -mi -crossC -reduce ig -residue 112 -model_dirs CUB/output/manuscript/112r_ig/iterNorm/ CUB/output/manuscript/112r_ig/iterNorm/ CUB/output/manuscript/112r_ig/iterNorm/ -log_dir CUB/output/manuscript/112r_ig/iterNorm/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
# python3 CUB/negTTI.py -ray -reduce ig -residue 112 -model_dirs CUB/output/manuscript/112r_ig/iterNorm/ CUB/output/manuscript/112r_ig/iterNorm/ CUB/output/manuscript/112r_ig/iterNorm/ -log_dir CUB/output/manuscript/112r_ig/iterNorm/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -log_dir CUB/output/manuscript/112r_ig/MI/ -reduce ig -ray -ckpt 1 -should_detach -residue 112 -disentangle club -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
# python3 CUB/inference.py -ray -mi -crossC -reduce ig -residue 112 -model_dirs CUB/output/manuscript/112r_ig/MI/ CUB/output/manuscript/112r_ig/MI/ CUB/output/manuscript/112r_ig/MI/ -log_dir CUB/output/manuscript/112r_ig/MI/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
# python3 CUB/negTTI.py -ray -reduce ig -residue 112 -model_dirs CUB/output/manuscript/112r_ig/MI/ CUB/output/manuscript/112r_ig/MI/ CUB/output/manuscript/112r_ig/MI/ -log_dir CUB/output/manuscript/112r_ig/MI/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -log_dir CUB/output/manuscript/112r_ig/crossC/ -reduce ig -ray -ckpt 1 -should_detach -residue 112 -disentangle crossC -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
# python3 CUB/inference.py -ray -mi -crossC -reduce ig -residue 112 -model_dirs CUB/output/manuscript/112r_ig/crossC/ CUB/output/manuscript/112r_ig/crossC/ CUB/output/manuscript/112r_ig/crossC/ -log_dir CUB/output/manuscript/112r_ig/crossC/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
# python3 CUB/negTTI.py -ray -reduce ig -residue 112 -model_dirs CUB/output/manuscript/112r_ig/crossC/ CUB/output/manuscript/112r_ig/crossC/ CUB/output/manuscript/112r_ig/crossC/ -log_dir CUB/output/manuscript/112r_ig/crossC/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 


###
# CUB importance reduction of concepts w/ 64 residuals---------------------------------------------------------
###

# python3 experiments.py cub Joint -log_dir CUB/output/manuscript/64r_ig/detached_c_r/ -reduce ig -ray -ckpt 1 -should_detach -residue 64 -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
# python3 CUB/inference.py -ray -mi -crossC -reduce ig -residue 64 -model_dirs CUB/output/manuscript/64r_ig/detached_c_r/ CUB/output/manuscript/64r_ig/detached_c_r/ CUB/output/manuscript/64r_ig/detached_c_r/ -log_dir CUB/output/manuscript/64r_ig/detached_c_r/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
# python3 CUB/negTTI.py -ray -reduce ig -residue 64 -model_dirs CUB/output/manuscript/64r_ig/detached_c_r/ CUB/output/manuscript/64r_ig/detached_c_r/ CUB/output/manuscript/64r_ig/detached_c_r/ -log_dir CUB/output/manuscript/64r_ig/detached_c_r/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -log_dir CUB/output/manuscript/64r_ig/iterNorm/ -reduce ig -ray -ckpt 1 -should_detach -residue 64 -disentangle IterNorm -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
# python3 CUB/inference.py -ray -mi -crossC -reduce ig -residue 64 -model_dirs CUB/output/manuscript/64r_ig/iterNorm/ CUB/output/manuscript/64r_ig/iterNorm/ CUB/output/manuscript/64r_ig/iterNorm/ -log_dir CUB/output/manuscript/64r_ig/iterNorm/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
# python3 CUB/negTTI.py -ray -reduce ig -residue 64 -model_dirs CUB/output/manuscript/64r_ig/iterNorm/ CUB/output/manuscript/64r_ig/iterNorm/ CUB/output/manuscript/64r_ig/iterNorm/ -log_dir CUB/output/manuscript/64r_ig/iterNorm/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -log_dir CUB/output/manuscript/64r_ig/MI/ -reduce ig -ray -ckpt 1 -should_detach -residue 64 -disentangle club -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
# python3 CUB/inference.py -ray -mi -crossC -reduce ig -residue 64 -model_dirs CUB/output/manuscript/64r_ig/MI/ CUB/output/manuscript/64r_ig/MI/ CUB/output/manuscript/64r_ig/MI/ -log_dir CUB/output/manuscript/64r_ig/MI/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
# python3 CUB/negTTI.py -ray -reduce ig -residue 64 -model_dirs CUB/output/manuscript/64r_ig/MI/ CUB/output/manuscript/64r_ig/MI/ CUB/output/manuscript/64r_ig/MI/ -log_dir CUB/output/manuscript/64r_ig/MI/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -log_dir CUB/output/manuscript/64r_ig/crossC/ -reduce ig -ray -ckpt 1 -should_detach -residue 64 -disentangle crossC -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
# python3 CUB/inference.py -ray -mi -crossC -reduce ig -residue 64 -model_dirs CUB/output/manuscript/64r_ig/crossC/ CUB/output/manuscript/64r_ig/crossC/ CUB/output/manuscript/64r_ig/crossC/ -log_dir CUB/output/manuscript/64r_ig/crossC/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
# python3 CUB/negTTI.py -ray -reduce ig -residue 64 -model_dirs CUB/output/manuscript/64r_ig/crossC/ CUB/output/manuscript/64r_ig/crossC/ CUB/output/manuscript/64r_ig/crossC/ -log_dir CUB/output/manuscript/64r_ig/crossC/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

###
# test
###
# python3 experiments.py cub Joint -subset 112 -log_dir CUB/output/manuscript_new/64r/detached_c_r/ -reduce random -ckpt 1 -should_detach -residue 64 -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end

###
# CUB random reduction of concepts w/ 32 residuals semi-supversied training ---------------------------------------------------------
###

# python3 experiments.py cub Joint -semi_supervise -log_dir CUB/output/manuscript_new/CUB/32r/concept_only/ -reduce random -ray -ckpt 1 -should_detach -residue 0 -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
python3 CUB/inference.py -ray -reduce random -model_dirs CUB/output/manuscript_new/CUB/32r/concept_only/ -log_dir CUB/output/manuscript_new/CUB/32r/concept_only/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -posTTI -ray -reduce random -model_dirs CUB/output/manuscript_new/CUB/32r/concept_only/ -log_dir CUB/output/manuscript_new/CUB/32r/concept_only/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -negTTI -ray -reduce random -model_dirs CUB/output/manuscript_new/CUB/32r/concept_only/ -log_dir CUB/output/manuscript_new/CUB/32r/concept_only/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -semi_supervise -log_dir CUB/output/manuscript_new/CUB/32r/detached_c_r/ -reduce random -ray -ckpt 1 -should_detach -residue 32 -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
python3 CUB/inference.py -ray -mi -crossC -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/detached_c_r/ -log_dir CUB/output/manuscript_new/CUB/32r/detached_c_r/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -posTTI -ray -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/detached_c_r/ -log_dir CUB/output/manuscript_new/CUB/32r/detached_c_r/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -negTTI -ray -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/detached_c_r/ -log_dir CUB/output/manuscript_new/CUB/32r/detached_c_r/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -semi_supervise -log_dir CUB/output/manuscript_new/CUB/32r/crossC/ -disentangle crossC -reduce random -ray -ckpt 1 -should_detach -residue 32 -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
python3 CUB/inference.py -ray -mi -crossC -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/crossC/ -log_dir CUB/output/manuscript_new/CUB/32r/crossC/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -posTTI -ray -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/crossC/ -log_dir CUB/output/manuscript_new/CUB/32r/crossC/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -negTTI -ray -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/crossC/ -log_dir CUB/output/manuscript_new/CUB/32r/crossC/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -semi_supervise -log_dir CUB/output/manuscript_new/CUB/32r/MI/ -disentangle club -reduce random -ray -ckpt 1 -should_detach -residue 32 -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
python3 CUB/inference.py -ray -mi -crossC -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/MI/ -log_dir CUB/output/manuscript_new/CUB/32r/MI/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -posTTI -ray -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/MI/ -log_dir CUB/output/manuscript_new/CUB/32r/MI/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -negTTI -ray -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/MI/ -log_dir CUB/output/manuscript_new/CUB/32r/MI/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

# python3 experiments.py cub Joint -semi_supervise -log_dir CUB/output/manuscript_new/CUB/32r/IterNorm/ -disentangle IterNorm -reduce random -ray -ckpt 1 -should_detach -residue 32 -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
python3 CUB/inference.py -ray -mi -crossC -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/IterNorm/ -log_dir CUB/output/manuscript_new/CUB/32r/IterNorm/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -posTTI -ray -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/IterNorm/ -log_dir CUB/output/manuscript_new/CUB/32r/IterNorm/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 
python3 CUB/newTTI.py -negTTI -ray -reduce random -residue 32 -model_dirs CUB/output/manuscript_new/CUB/32r/IterNorm/ -log_dir CUB/output/manuscript_new/CUB/32r/IterNorm/ -eval_data test -use_attr -data_dir CUB_processed/class_attr_data_10 -feature_group_results 

###
# compute feature importance
###
python3 CUB/feature_attribution.py -model_dirs CUB/output/manuscript_new/CUB/32r/concept_only/112/best_model_1.pt -log_dir CUB/output/manuscript_new/CUB/ -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -feature_group_results 


###
# HAM10k
###
# python3 experiments.py ham10k Joint -log_dir CUB/output/manuscript_new/ham10k/64r/detached_c_r/ -reduce random -ray -ckpt 1 -semi_supervise -should_detach -residue 4 -e 100 -optimizer Adam -pretrained -use_aux -use_attr -weighted_loss multiple -n_attributes 8 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end -concept-bank post_hoc_cbm/concept_bank/derm7pt_ham10000_inception_0.01_50.pkl -backbone-name ham10000_inception
