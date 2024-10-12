# Concept Bottleneck Models - CUB Dataset
## Dataset preprocessing
1) Download the [official CUB dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) (`CUB_200_2011`), processed CUB data (`CUB_processed`), places365 dataset (`places365`) and pretrained Inception V3 models (`pretrained`) from [Codalab worksheet](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2).   

OR) You can get `CUB_processed` from Step 1. above with the following steps
1) Run `data_processing.py` to obtain train/ val/ test splits as well as to extract all relevant task and concept metadata into pickle files. 
2) Run `generate_new_data.py` to obtain other versions of training data (class-level attributes, few-shot training, etc.) from the metadata.
## Experiments
1) Update the paths (e.g. `BASE_DIR`, `-log_dir`, `-out_dir`, `--model_path`) in the scripts (where applicable) to point to your dataset and outputs.