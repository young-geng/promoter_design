# Designing Cell-Type-Specific Promoter Sequences with Model-Based Optimization

This repository provides a flexible, data-efficient, and scalable workflow for designing cell-type-specific promoter sequences by leveraging transfer learning [1] and conservative model-based optimization [2]. Please cite the following paper if you use our code or data:
```bibtex
@inproceedings{
    promoter_design_reddy_geng_herschl_2024,
    title={Designing Cell-Type-Specific Promoter Sequences Using Conservative Model-Based Optimization},
    author={Aniketh Janardhan Reddy and Xinyang Geng and Michael H Herschl and Sathvik Kolli and Aviral Kumar and Patrick D Hsu and Sergey Levine and Nilah M Ioannidis},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=F8DWffLkYG}
}
```

## Prerequisites

1. We use Weights and Biases for logging. Please set up your environment by following the instructions [here](https://docs.wandb.ai/quickstart).

## Workflow

We run our workflow to design cell-type-specific promoter sequences for Jurkat, K562, and THP1 cell lines. Here, we describe the steps to run the workflow.

### Step 1: Pretraining models on existing data from massively parallel reporter assays (MPRAs)

To improve the data efficiency of our workflow, we pretrain models on existing data from MPRAs. We follow Reddy, Herschl, and Geng et al. (2023) [1] and pretrain on data from SuRE [3] and Sharpr-MPRA [4]:

1. We provide the processed SuRE data at [this link](https://huggingface.co/datasets/anikethjr/promoter_design/tree/main). Please download the data and place the SuRE directory in the `data` directory. Processing was performed using the [code](https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/SuRE.py) provided by Reddy, Herschl, and Geng et al. (2023) [1].
2. The processed Sharpr-MPRA data was made available by Movva et al. [5]. Please download the data from the following links and place them in the `data/Sharpr_MPRA` directory:
    - [train.hdf5](https://mitra.stanford.edu/kundaje/projects/mpra/data/train.hdf5)
    - [valid.hdf5](https://mitra.stanford.edu/kundaje/projects/mpra/data/valid.hdf5)
    - [test.hdf5](https://mitra.stanford.edu/kundaje/projects/mpra/data/test.hdf5)
3. Run the following command to format the data for training:
    ```bash
    python -m promoter_design.workflow.process_pretrain_data --sure_dir data/SuRE --mpra_dir data/Sharpr_MPRA --output_file data/all_data.pkl
    ```
4. Finally, run the following commands to pretrain an MTLucifer model [1] (we performed training on an NVIDIA DGX A100 with 8 A100 GPUs and it took approximately 3.5 hours):
    ```bash
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
    mkdir -p models/pretrain

    python -m promoter_design.workflow.pretrain_main \
        --total_steps=20000 \
        --eval_freq=200 \
        --eval_steps=100 \
        --save_model=True \
        --accumulate_gradient_steps=1 \
        --lr=1e-4 \
        --weight_decay=3e-3 \
        --k562_loss_weight=1.0 \
        --hepg2_loss_weight=1.0 \
        --mpra_loss_weight=1.0 \
        --pretrain_network.backbone.embedding_dim=1024 \
        --pretrain_network.backbone.num_heads=8 \
        --pretrain_network.backbone.transformer_blocks=5 \
        --train_data.path="data/all_data.pkl" \
        --train_data.split='train' \
        --train_data.batch_size=448 \
        --eval_data.path="data/all_data.pkl" \
        --eval_data.split='val' \
        --eval_data.batch_size=448 \
        --logger.output_dir="models/pretrain" \
        --logger.online=True \
        --logger.prefix='promoter_design' \
        --logger.project="pretrain"
    ```

### Step 2: Fine-tuning models on data from targeted cell lines and generating sequences

We fine-tune the pretrained models on data from Jurkat, K562, and THP1 cell lines and generate sequences using conservative model-based optimization [2]:

1. The `data/finetuning_data.csv` file contains the data for fine-tuning the models, from Reddy, Herschl, and Geng et al. (2023) [1]. Run the following command to format the data for training:
    ```bash
    python -m promoter_design.workflow.process_finetune_data --input_file data/finetuning_data.csv --output_file data/finetune_data.pkl
    ```

2. Run the following commands to fine-tune the pretrained models and generate candidate sequences. The commands sequentially fine-tune 6 different models, each with a different level of conservatism. Then, it performs sequence design for each of the three cells using the fine-tuned models. Again, we performed training on an NVIDIA DGX A100 with 8 A100 GPUs and each task took approximately 2 hours:
    ```bash
    mkdir -p models/finetune
    export PRETRAINED_MODEL_PATH=$(find models/pretrain -name best_params.pkl)

    parallel -j1 --linebuffer \
        python -m promoter_design.workflow.finetune_main \
            --total_steps=200 \
            --log_freq=1 \
            --eval_freq=25 \
            --val_steps=3 \
            --test_steps=7 \
            --save_model=True \
            --load_pretrained="$PRETRAINED_MODEL_PATH" \
            --lr=5e-5 \
            --lr_warmup_steps=10 \
            --weight_decay=3e-3 \
            --use_coms_loss=True \
            --sequence_optimizer.total_rounds={1} \
            --sequence_optimizer.mutation_steps=0 \
            --sequence_optimizer.gd_steps={2} \
            --sequence_optimizer.gd_step_size=0.5 \
            --coms_loss_weight={3} \
            --expression_objective.thp1_exp_multiplier={4} \
            --expression_objective.type={5} \
            --expression_objective.linear_thp1_weight={6} \
            --train_data.path="data/finetune_data.pkl" \
            --train_data.split='train' \
            --train_data.batch_size=512 \
            --val_data.path="data/finetune_data.pkl" \
            --val_data.split='val' \
            --val_data.sequential_sample=True \
            --val_data.batch_size=512 \
            --test_data.path="data/finetune_data.pkl" \
            --test_data.split='test' \
            --test_data.sequential_sample=True \
            --test_data.batch_size=512 \
            --generation_data.path="data/finetune_data.pkl" \
            --generation_data.split='all' \
            --generation_data.sequential_sample=True \
            --generation_data.batch_size=128 \
            --logger.output_dir="models/finetune" \
            --logger.online=True \
            --logger.prefix='promoter_design' \
            --logger.project="finetune" \
        ::: 1 \
        ::: 100 \
        ::: 0.0 0.0003 0.001 0.003 0.01 0.03 \
        ::: 1.0 \
        ::: 'linear' \
        ::: 1.5
    ```

### Step 3: Train ensemble for final sequence selection

From the previous step, we obtain many candidate sequences for each cell line. Our final sequence selection algorithm uses an ensemble of models to evaluate the sequences. We train it as follows:

1. The ensemble is trained using a different split of the data used for fine-tuning. Run the following command to format the data for training:
    ```bash
    python -m promoter_design.workflow.process_finetune_data --input_file data/finetuning_data_ensemble.csv --output_file data/finetune_data_ensemble.pkl
    ```

2. Run the following commands to train the ensemble. The commands sequentially train 36 different models, each with a different output head configuration. We performed training on an NVIDIA DGX A100 with 8 A100 GPUs:
    ```bash
    mkdir -p models/ensemble
    export PRETRAINED_MODEL_PATH=$(find models/pretrain -name best_params.pkl)

    parallel -j1 --linebuffer \
        python -m promoter_design.workflow.finetune_main \
            --seed=34 \
            --total_steps=250 \
            --log_freq=1 \
            --eval_freq=25 \
            --val_steps=3 \
            --test_steps=7 \
            --save_model=True \
            --load_pretrained="$PRETRAINED_MODEL_PATH" \
            --lr=5e-5 \
            --lr_warmup_steps=10 \
            --weight_decay=3e-3 \
            --use_coms_loss=False \
            --generate_sequences=False \
            --finetune_network.output_head_num_layers={1} \
            --finetune_network.output_head_hidden_dim={2} \
            --finetune_network.output_head_activation={3} \
            --train_data.path="$PROJECT_HOME/data/finetune_data_ensemble.pkl" \
            --train_data.split='train' \
            --train_data.batch_size=512 \
            --val_data.path="$PROJECT_HOME/data/finetune_data_ensemble.pkl" \
            --val_data.split='val' \
            --val_data.sequential_sample=True \
            --val_data.batch_size=512 \
            --test_data.path="$PROJECT_HOME/data/finetune_data_ensemble.pkl" \
            --test_data.split='test' \
            --test_data.sequential_sample=True \
            --test_data.batch_size=512 \
            --logger.output_dir="$OUTPUT_DIR/$EXP_NAME" \
            --logger.online=True \
            --logger.prefix='promoter_design' \
            --logger.project="ensemble" \
        ::: 2 4 8 \
        ::: 512 1024 2048 \
        ::: 'tanh' 'gelu' 'relu' 'silu'
    ```

## References:
1. Reddy, Aniketh Janardhan, Michael H. Herschl, Xinyang Geng, Sathvik Kolli, Amy X. Lu, Aviral Kumar, Patrick D. Hsu, Sergey Levine, and Nilah M. Ioannidis. "Strategies for effectively modelling promoter-driven gene expression using transfer learning." bioRxiv (2023).
2. Trabucco, Brandon, Aviral Kumar, Xinyang Geng, and Sergey Levine. "Conservative objective models for effective offline model-based optimization." In International Conference on Machine Learning, pp. 10358-10368. PMLR, 2021.
3. van Arensbergen, Joris, Ludo Pagie, Vincent D. FitzPatrick, Marcel de Haas, Marijke P. Baltissen, Federico Comoglio, Robin H. van der Weide et al. "High-throughput identification of human SNPs affecting regulatory element activity." Nature genetics 51, no. 7 (2019): 1160-1169.
4. Ernst, Jason, Alexandre Melnikov, Xiaolan Zhang, Li Wang, Peter Rogov, Tarjei S. Mikkelsen, and Manolis Kellis. "Genome-scale high-resolution mapping of activating and repressive nucleotides in regulatory regions." Nature biotechnology 34, no. 11 (2016): 1180-1190.
5. Movva, Rajiv, Peyton Greenside, Georgi K. Marinov, Surag Nair, Avanti Shrikumar, and Anshul Kundaje. "Deciphering regulatory DNA sequences and noncoding genetic variants using neural network models of massively parallel reporter assays." PLoS One 14, no. 6 (2019): e0218073.