# Source Code  for "Towards a Real-World Aligned Benchmark for Unlearning in Recommender Systems"

## Recommender system code source

The recommender system code structure is based on from [**A‑Next‑Basket‑Recommendation‑Reality‑Check**](https://github.com/liming-7/A-Next-Basket-Recommendation-Reality-Check/).

## Hardware Info

Experiments ran using:

* CPU Memory: 20GB
* GPU: NVIDIA A100 80GB PCIe GPU

The experiments mainly rely on the GPU. For the CPU it is only important to have sufficient memory.

## Project Setup

To build an apptainer container for running the project you need the apptainer package installed. We used apptainer v1.4.1. You can install it on Ubuntu like this:

                cd <PATH_FOR_APPTAINER_REPOSITORY>
                git clone https://github.com/apptainer/apptainer.git
                cd apptainer
                git checkout v1.4.1
                ./mconfig --prefix=/usr/local --with-suid
                make -C builddir -j$(nproc)
                sudo make -C builddir install

Now go into the sets2sets directory

                cd <PATH_FOR_towards-unlearning-in-recsys>/methods/sets2sets/

With apptainer you can build the container using the following code:

                sudo apptainer build --force container.sif container.def

For fast execution of the container install the following packages: `squashfuse fuse2fs gocryptfs`
On Ubuntu this is possible with:

                sudo apt-get install squashfuse fuse2fs gocryptfs

Alternatively you can use Python3.12, create a venv, install the requirements.txt manually, and run the files with python instead of from the container. The `container.def` file contains information on how dependencies are installed inside the container which can be done similarly on a local machine. The execution of python scripts is the same as the apptainer calls without the prefix `apptainer run --nv --bind ./models/:/opt/towards-unlearning-in-recsys/methods/sets2sets/models/ container.sif`.

## Reproduce results

Go to the Sets2sets directory first:

                cd <PATH_TO_towards-unlearning-in-recsys>/methods/sets2sets/

To train a model on the original instacart dataset use:

                apptainer run --nv --bind ./models/:/opt/towards-unlearning-in-recsys/methods/sets2sets/models/ \
                    container.sif python sets2sets_new.py instacart 0 10 1 <SEED> 1 1

the trained model will be saved to \<PATH_TO_towards-unlearning-in-recsys\>/methods/sets2sets/models/ on your local machine with SEED being the random seed. A Sets2sets model consists of a encoder and a decoder, saved in a separate file. We provide one example model trained on Instacart with seed 2 in the <PATH_TO_towards-unlearning-in-recsys>/methods/sets2sets/models/ folder.

Retrained models can be achieved like this:

                apptainer run --nv --bind ./models/:/opt/towards-unlearning-in-recsys/methods/sets2sets/models/ \
                    container.sif python unlearn_sets2sets.py \
                    --temporal_split \
                    --method sensitive \
                    --sensitive_category <CATEGORY> \
                    --unlearning_fraction 0.001 
                    --seed <SEED> \
                    --training 1 \
                    --retrain_checkpoint_idx_to_match <IDX> \
                    --dataset instacart

with values of --sensitive_category in ["meat", "alcohol", baby"], --seed being the random seed (we used [2, 3, 5, 7 ,11]), and --retrain_checkpoint_idx_to_match in [0, 1, 2, 3] where retrain_checkpoint_idx_to_match $i$ uses $(i + 1)/4$ of the total unlearning requests in the current setting. You can also use the --LOCAL flag to get a tqdm progress bar.

To unlearn a trained model using an unlearning algorithm use the following command:

                apptainer run --nv --bind ./models/:/opt/towards-unlearning-in-recsys/methods/sets2sets/models/ \
                    container.sif python unlearn_sets2sets.py \
                    --temporal_split \
                    --method sensitive \
                    --sensitive_category <CATEGORY> \
                    --unlearning_fraction 0.001 \
                    --seed <SEED> \
                    --training 2 \
                    --unlearning_algorithm <UNLEARNING_ALGORITHM> \
                    --dataset instacart

with values of --sensitive_category in ["meat", "alcohol", baby"], --seed being the random seed (we used [2, 3, 5, 7 ,11]), and --unlearning_algorithm in ["kookmin", "fanchuan", "scif"]. You can also use the --LOCAL flag to get a tqdm progress bar. At the end of unlearning metrics will be computed for the models at the 4 checkpoints (1/4, 2/4, 3/4, 4/4 of requests). For the metric calculations the corresponding retrained models are also needed.

When models are trained/retrained/unlearned you can also evaluate models using `evaluate_unlearning.py --category <CATEGORY>` to evaluate all models for category CATEGORY. Setting category to "all" will evaluate all models.


## Structure
* preprocess: contains the script of dataset preprocessing.
* dataset: contains the .csv format dataset after preprocessing.
* jsondata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored seperately.
* mergedata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored together.
* methods: contains the source code of different NBR methods and the original url repository link of these methods.
* keyset_fold.py: splits the datasets across users for train/validate/test.
* evaluation: scripts for evaluation.
    * metrics.py: the general metrics.
    * performance_gain.py: evaluate the contribution of repetition and exploration.
    * model_performance.py: evaluate the baskets' rep/expl ratio, and the recall, phr performance w.r.t. repetition and exploration.
* unlearning_data: contains deterministically created forget sets depending on seed, category and unlearning_fraction, which can be recreated using methods/sets2sets/create_unlearning_sets.py



### Dataset description:

* jsondata:

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }

> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}


## Guidelines for each method
Original code for models implemented can be found here:
* DREAM: https://github.com/yihong-chen/DREAM
* BEACON: https://github.com/PreferredAI/beacon
* CLEA: https://github.com/QYQ-bot/CLEA/tree/main/CLEA
* Sets2Sets: https://github.com/HaojiHu/Sets2Sets
* DNNTSP: https://github.com/yule-BUAA/DNNTSP
* TIFUKNN: https://github.com/HaojiHu/TIFUKNN
* UP-CF@r: https://github.com/MayloIFERR/RACF
