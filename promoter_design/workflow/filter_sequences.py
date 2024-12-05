import numpy as np
import os
import mlxu
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

np.random.seed(97)

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    designed_sequences_file='',
    dataset_file='',
    output_file='',
    expression_percentile_thres=90,
    design_method='COMs',
)

m = ["A", "C", "G", "T"]
def id2seq(ids):
    return "".join([m[i] for i in ids])

def main(argv):
    assert FLAGS.designed_sequences_file != ''
    assert FLAGS.dataset_file != ''
    assert FLAGS.output_file != ''

    designed_sequences = mlxu.load_pickle(FLAGS.designed_sequences_file)
    dataset_sequences = mlxu.load_pickle(FLAGS.dataset_file)

    expression_percentile_thres = FLAGS.expression_percentile_thres
    
    # process designed sequences
    designed_sequences_df = {}
    designed_sequences_df["design_method"] = []
    designed_sequences_df["designed_for"] = []
    designed_sequences_df["original_sequence"] = []
    designed_sequences_df["sequence"] = []
    designed_sequences_df["provenance"] = []
    designed_sequences_df["coms_alpha"] = []
    designed_sequences_df["diversity_loss_coef"] = []
    designed_sequences_df["entropy_loss_coef"] = []
    designed_sequences_df["base_entropy_loss_coef"] = []

    for cell in ["Jurkat", "K562", "THP1"]:    
        designed_sequences_df[cell + "_ensemble_mean"] = []
        designed_sequences_df[cell + "_ensemble_std"] = []
        designed_sequences_df[cell + "_all_ensemble_preds"] = []
        designed_sequences_df[cell + "_design_model"] = []
        designed_sequences_df[cell + "_measured"] = []

    if FLAGS.design_method == "COMs":
        ori_sequences = [id2seq(i) for i in designed_sequences["original_seq"]]
    elif FLAGS.design_method == "DENs":
        ori_sequences = [None for i in designed_sequences["sequence"]]

    for cell in ["jurkat", "k562", "thp1"]:
        print(cell)
        sequences = [id2seq(i) for i in designed_sequences[f"{cell}_optimized_seq"]]
        
        corrected_cell_name = cell.upper()
        if cell == "jurkat":
            corrected_cell_name = "Jurkat"
        
        designed_sequences_df["designed_for"] += [corrected_cell_name for i in range(len(sequences))]
        designed_sequences_df["original_sequence"] += ori_sequences.copy()
        designed_sequences_df["sequence"] += ["".join(i) for i in sequences]

        designed_sequences_df["design_method"] += [FLAGS.design_method for i in sequences]
        if FLAGS.design_method == "COMs":
            designed_sequences_df["coms_alpha"] += list(designed_sequences["coms_loss_weight"])
            designed_sequences_df["provenance"] += [None for i in sequences]
            designed_sequences_df["diversity_loss_coef"] += [None for i in sequences]
            designed_sequences_df["entropy_loss_coef"] += [None for i in sequences]
            designed_sequences_df["base_entropy_loss_coef"] += [None for i in sequences]
        elif FLAGS.design_method == "DENs":
            designed_sequences_df["coms_alpha"] += [float(i.split("_")[1]) for i in designed_sequences["design_model"]]
            designed_sequences_df["provenance"] += [None for i in sequences]
            designed_sequences_df["diversity_loss_coef"] += list(designed_sequences["diversity_loss_coef"])
            designed_sequences_df["entropy_loss_coef"] += list(designed_sequences["entropy_loss_coef"])
            designed_sequences_df["base_entropy_loss_coef"] += list(designed_sequences["base_entropy_loss_coef"])
        
        for cell2 in ["Jurkat", "K562", "THP1"]:
            designed_sequences_df[cell2 + "_ensemble_mean"] += list(designed_sequences[f"ensemble_{cell}_optimized_seq_{cell2.lower()}_pred"].mean(axis=0))
            designed_sequences_df[cell2 + "_ensemble_std"] += list(designed_sequences[f"ensemble_{cell}_optimized_seq_{cell2.lower()}_pred"].std(axis=0))
            designed_sequences_df[cell2 + "_all_ensemble_preds"] += [designed_sequences[f"ensemble_{cell}_optimized_seq_{cell2.lower()}_pred"]]
            designed_sequences_df[cell2 + "_design_model"] += list(designed_sequences[f"{cell}_opt_seq_{cell2.lower()}_pred"])
            designed_sequences_df[cell2 + "_measured"] += [None for i in sequences]

    cp = {}
    for cell2 in ["Jurkat", "K562", "THP1"]:
        designed_sequences_df[cell2 + "_all_ensemble_preds"] = np.hstack(designed_sequences_df[cell2 + "_all_ensemble_preds"]).T
        cp[cell2] = designed_sequences_df[cell2 + "_all_ensemble_preds"].copy()
        designed_sequences_df.pop(cell2 + "_all_ensemble_preds")

    designed_sequences_df = pd.DataFrame(designed_sequences_df)

    for cell2 in ["Jurkat", "K562", "THP1"]:
        designed_sequences_df[cell2 + "_all_ensemble_preds"] = [cp[cell2][i, :] for i in range(designed_sequences_df.shape[0])]

    # process original dataset sequences
    dataset_df = {}
    dataset_df["design_method"] = []
    dataset_df["designed_for"] = []
    dataset_df["original_sequence"] = []
    dataset_df["sequence"] = []
    dataset_df["coms_alpha"] = []
    dataset_df["provenance"] = []
    dataset_df["diversity_loss_coef"] = []
    dataset_df["entropy_loss_coef"] = []
    dataset_df["base_entropy_loss_coef"] = []

    for cell in ["Jurkat", "K562", "THP1"]:    
        dataset_df[cell + "_ensemble_mean"] = []
        dataset_df[cell + "_ensemble_std"] = []
        dataset_df[cell + "_all_ensemble_preds"] = []
        dataset_df[cell + "_design_model"] = []
        dataset_df[cell + "_measured"] = []

    dataset_df["designed_for"] += [None for i in dataset_sequences["sequences"]]
    dataset_df["design_method"] += ["Dataset" for i in dataset_sequences["sequences"]]
    dataset_df["original_sequence"] += [None for i in dataset_sequences["sequences"]]

    dataset_df["sequence"] += [id2seq(i) for i in dataset_sequences["sequences"]]

    dataset_df["provenance"] += [None for i in dataset_sequences["sequences"]]
    dataset_df["coms_alpha"] += [None for i in dataset_sequences["sequences"]]
    dataset_df["diversity_loss_coef"] += [None for i in dataset_sequences["sequences"]]
    dataset_df["entropy_loss_coef"] += [None for i in dataset_sequences["sequences"]]
    dataset_df["base_entropy_loss_coef"] += [None for i in dataset_sequences["sequences"]]

    for cell2 in ["Jurkat", "K562", "THP1"]:
        dataset_df[cell2 + "_ensemble_mean"] += list(dataset_sequences[f"ensemble_sequences_{cell2.lower()}_pred"].mean(axis=0))
        dataset_df[cell2 + "_ensemble_std"] += list(dataset_sequences[f"ensemble_sequences_{cell2.lower()}_pred"].std(axis=0))
        dataset_df[cell2 + "_all_ensemble_preds"] += [dataset_sequences[f"ensemble_sequences_{cell2.lower()}_pred"]]
        dataset_df[cell2 + "_design_model"] += [None for i in dataset_sequences["sequences"]]
        dataset_df[cell2 + "_measured"] += list(dataset_sequences[f"{cell2.lower()}_output"])
        
    cp = {}
    for cell2 in ["Jurkat", "K562", "THP1"]:
        dataset_df[cell2 + "_all_ensemble_preds"] = np.hstack(dataset_df[cell2 + "_all_ensemble_preds"]).T
        cp[cell2] = dataset_df[cell2 + "_all_ensemble_preds"].copy()
        dataset_df.pop(cell2 + "_all_ensemble_preds")

    dataset_df = pd.DataFrame(dataset_df)

    for cell2 in ["Jurkat", "K562", "THP1"]:
        dataset_df[cell2 + "_all_ensemble_preds"] = [cp[cell2][i, :] for i in range(dataset_df.shape[0])]
        
    # filter sequences based on predicted expression and sort by differential expression
    designed_sequences_df["filter_out"] = False
    designed_sequences_df["diff_exp"] = 0.0
    for i, cell1 in enumerate(["Jurkat", "K562", "THP1"]):
        print(cell1)
        expression_percentile_thres_val = np.percentile(dataset_df[f"{cell1}_ensemble_mean"], expression_percentile_thres)
        print(f"Expression threshold = {expression_percentile_thres_val}")
        
        ori_num_seqs = (designed_sequences_df['designed_for'] == cell1).sum()
        print(f"We had {ori_num_seqs} {FLAGS.design_method} sequences")
        
        designed_sequences_df.loc[(designed_sequences_df["designed_for"] == cell1) & 
                    (designed_sequences_df[f"{cell1}_ensemble_mean"] < expression_percentile_thres_val), "filter_out"] = True
        
        left_after_exp_thres = ((designed_sequences_df['designed_for'] == cell1) & ~designed_sequences_df['filter_out']).sum()
        print(f"After filtering based on the expression threshold, we have {left_after_exp_thres} {FLAGS.design_method} designed sequences")
        
        designed_sequences_df.loc[designed_sequences_df["designed_for"] == cell1, 
                    "diff_exp"] = designed_sequences_df.loc[designed_sequences_df["designed_for"] == cell1, 
                                            f"{cell1}_ensemble_mean"]
        
        for j, cell2 in enumerate(["Jurkat", "K562", "THP1"]):
            if cell1 == cell2:
                continue
            designed_sequences_df.loc[(designed_sequences_df["designed_for"] == cell1) & 
                        (designed_sequences_df[f"{cell1}_ensemble_mean"] < designed_sequences_df[f"{cell2}_ensemble_mean"]), "filter_out"] = True
            
            designed_sequences_df.loc[designed_sequences_df["designed_for"] == cell1, 
                    "diff_exp"] -= (designed_sequences_df.loc[designed_sequences_df["designed_for"] == cell1, 
                                            f"{cell2}_ensemble_mean"] * 0.5)
            
        left_after_DE_filt = ((designed_sequences_df['designed_for'] == cell1) & ~designed_sequences_df['filter_out']).sum()
        print(f"After filtering based on the having non-zero DE, we have {left_after_DE_filt} {FLAGS.design_method} designed sequences")

    # actual filtering
    designed_sequences_df = designed_sequences_df[~designed_sequences_df["filter_out"]].sort_values(by="diff_exp").reset_index(drop=True)
    
    # save the filtered sequences
    designed_sequences_df.to_parquet(FLAGS.output_file)
    

if __name__ == '__main__':
    mlxu.run(main)