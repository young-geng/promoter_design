# Designing Cell-Type-Specific Promoter Sequences with Model-Based Optimization

This repository provides a flexible, data-efficient, and scalable workflow for designing cell-type-specific promoter sequences by leveraging transfer learning [1] and conservative model-based optimization [2]. Please cite the following paper if you use our code or data:
```bibtex
@article {promoter_design_reddy_geng_herschl_2024,
	author = {Reddy, Aniketh Janardhan and Geng, Xinyang and Herschl, Michael H. and Kolli, Sathvik and Kumar, Aviral and Hsu, Patrick D. and Levine, Sergey and Ioannidis, Nilah M.},
	title = {Designing Cell-Type-Specific Promoter Sequences Using Conservative Model-Based Optimization},
	elocation-id = {2024.06.23.600232},
	year = {2024},
	doi = {10.1101/2024.06.23.600232},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/06/23/2024.06.23.600232},
	eprint = {https://www.biorxiv.org/content/early/2024/06/23/2024.06.23.600232.full.pdf},
	journal = {bioRxiv}
}
```

## Prerequisites

1. We use Weights and Biases for logging. Please set up your environment by following the instructions [here](https://docs.wandb.ai/quickstart).

## Workflow

We run our workflow to design cell-type-specific promoter sequences for Jurkat, K562, and THP1 cell lines. Here, we describe the steps to run the workflow.

### Step 1: Pretraining models on existing data from massively parallel reporter assays (MPRAs)

To improve the data efficiency of our workflow, we pretrain models on existing data from MPRAs. We follow Reddy, Herschl, and Geng et al. (2023) [1] and pretrain on data from SuRE [3] and Sharpr-MPRA [4]:

1. We provide the processed SuRE data in the `data` directory. Processing was performed using the [code](https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/SuRE.py) provided by Reddy, Herschl, and Geng et al. (2023) [1].
2. The processed Sharpr-MPRA data was made available by Movva et al. [5]. Please download the data from the following links and place them in the `data/Sharpr_MPRA` directory:
    - [train.hdf5](https://mitra.stanford.edu/kundaje/projects/mpra/data/train.hdf5)
    - [valid.hdf5](https://mitra.stanford.edu/kundaje/projects/mpra/data/valid.hdf5)
    - [test.hdf5](https://mitra.stanford.edu/kundaje/projects/mpra/data/test.hdf5)
3. Run the following command to format the data for training:
    ```bash
    python -m promoter_design.promoter_modelling.process_pretrain_data --sure_dir data/SuRE --mpra_dir data/Sharpr_MPRA --output_file data/all_data.pkl
    ```
4. Finally, run the following commands to pretrain an MTLucifer model [1] (we performed training on an NVIDIA DGX A100 with 8 A100 GPUs and it took approximately 3.5 hours):
    ```bash
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
    mkdir -p models/pretrain

    python -m promoter_design.promoter_modelling.pretrain_main \
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
        --logger.project="pretrain" \
    ```

## References:
1. Reddy, Aniketh Janardhan, Michael H. Herschl, Xinyang Geng, Sathvik Kolli, Amy X. Lu, Aviral Kumar, Patrick D. Hsu, Sergey Levine, and Nilah M. Ioannidis. "Strategies for effectively modelling promoter-driven gene expression using transfer learning." bioRxiv (2023).
2. Trabucco, Brandon, Aviral Kumar, Xinyang Geng, and Sergey Levine. "Conservative objective models for effective offline model-based optimization." In International Conference on Machine Learning, pp. 10358-10368. PMLR, 2021.
3. van Arensbergen, Joris, Ludo Pagie, Vincent D. FitzPatrick, Marcel de Haas, Marijke P. Baltissen, Federico Comoglio, Robin H. van der Weide et al. "High-throughput identification of human SNPs affecting regulatory element activity." Nature genetics 51, no. 7 (2019): 1160-1169.
4. Ernst, Jason, Alexandre Melnikov, Xiaolan Zhang, Li Wang, Peter Rogov, Tarjei S. Mikkelsen, and Manolis Kellis. "Genome-scale high-resolution mapping of activating and repressive nucleotides in regulatory regions." Nature biotechnology 34, no. 11 (2016): 1180-1190.
5. Movva, Rajiv, Peyton Greenside, Georgi K. Marinov, Surag Nair, Avanti Shrikumar, and Anshul Kundaje. "Deciphering regulatory DNA sequences and noncoding genetic variants using neural network models of massively parallel reporter assays." PLoS One 14, no. 6 (2019): e0218073.