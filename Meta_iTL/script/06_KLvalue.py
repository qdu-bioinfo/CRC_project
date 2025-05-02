import os
import sys
import pandas as pd
import numpy as np
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'
figure_path = sys.path[1] + '\\Result\\figures\\'
param_path=sys.path[1] + '\\Result\\param\\'
_eps = 1e-10
def kl_divergence_matrix(A, B):
    P = A[:, None, :] + _eps   # shape (n,1,d)
    Q = B[None, :, :] + _eps   # shape (1,m,d)
    return np.sum(P * np.log(P / Q), axis=2)  # shape (n,m)
def js_divergence_matrix(A, B):
    P = A[:, None, :] + _eps    # shape (n,1,d)
    Q = B[None, :, :] + _eps    # shape (1,m,d)
    M = 0.5 * (P + Q)           # shape (n,m,d)

    kl_PM = np.sum(P * np.log(P / M), axis=2)  # D_KL(P||M), shape (n,m)
    kl_QM = np.sum(Q * np.log(Q / M), axis=2)  # D_KL(Q||M), shape (n,m)

    return 0.5 * (kl_PM + kl_QM)
def compute_kl_between_distributions(A, B):
    return kl_divergence_matrix(A, B).mean()
def compute_js_between_distributions(A, B):
    return js_divergence_matrix(A, B).mean()
def bootstrap_js(data1, data2, n_repeat=100):
    n1, n2 = data1.shape[0], data2.shape[0]
    boot_means = np.empty(n_repeat, dtype=np.float64)
    for i in range(n_repeat):
        idx1 = np.random.randint(0, n1, size=n1)
        idx2 = np.random.randint(0, n2, size=n2)
        A = data1[idx1]
        B = data2[idx2]
        js_mat = js_divergence_matrix(A, B)
        boot_means[i] = js_mat.mean()
    return boot_means.mean(), np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)

for ratio in ["S0.2","S0.3","S0.4","S0.5","S0.6"]:
    # Create a directory for the current ratio
    ratio_dir = f"{figure_path}/JS_KL/{ratio}/"
    os.makedirs(ratio_dir, exist_ok=True)

    # Initialize an empty DataFrame to store results for this ratio
    all_results = pd.DataFrame()

    # target_study = "CHN_SH-CRC-4"
    # source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-2_US-CRC-3"

    target_study="CHN_WF-CRC"
    source_study="FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-4_CHN_SH-CRC-2"

    # Load the target dataset
    target = pd.read_csv(
        f'{data_path}/{source_study}_{target_study}/{target_study}_{ratio}_test.csv',
        index_col=0
    )
    target = target[target['Group'].isin(["ADA", "CTR"])]
    target = target.drop(columns=['Study'], errors='ignore')

    # Load the feature dataset
    Meta_iTL = pd.read_csv(
        f'{data_path}/{source_study}_{target_study}/filtered_{source_study}_{target_study}_{ratio}_train.csv',
        index_col=0
    )
    Meta_iTL = Meta_iTL[Meta_iTL['Group'].isin(["ADA", "CTR"])]
    Meta_iTL = Meta_iTL.drop(columns=['Study'], errors='ignore')

    feature = pd.read_csv(f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_optimal.csv")
    feature_target = pd.read_csv(f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/target_optimal.csv")
    frequency_feature_name2 = "Target_" + source_study + "_" + target_study + "_synergistic"
    finally_rf_optimal_feature2 = feature_target[frequency_feature_name2].dropna().tolist()
    frequency_feature_name = source_study + "_" + target_study + "_synergistic"
    finally_rf_optimal_feature = feature[frequency_feature_name].dropna().tolist()
    #optimal_features_set = list(dict.fromkeys(finally_rf_optimal_feature2 + finally_rf_optimal_feature))
    optimal_features_set = list(dict.fromkeys(finally_rf_optimal_feature))
    Meta_iTL_feature =optimal_features_set

    feature_finally = pd.read_csv(f"{feature_path}raw/{source_study}_optimal.csv")
    Raw_feature=pd.Series(feature_finally[f"{source_study}_synergistic"]).dropna().tolist()
    # Load the original dataset
    meta_all = pd.read_csv(data_path+"/meta.csv")
    meta_all = meta_all[meta_all['Study'].isin(source_study.split("_")) & meta_all['Group'].isin(["ADA", "CTR"])]
    feature_all = pd.read_csv(data_path+"/feature_rare_CTR_ADA.csv",sep='\t', index_col=0)
    feature_all = np.log(feature_all + 1)
    feature_all = feature_all.loc[meta_all['Sample_ID']]
    # Combine data
    def normalize(row):
        return row / np.sum(row)
    #Target vs Meta_iTL
    target['batch'] = "target"
    Meta_iTL['batch'] = "TL"
    combined_df1 = pd.concat([target[["batch","Group"]+optimal_features_set], Meta_iTL[["batch","Group"]+optimal_features_set]], axis=0, ignore_index=True)
    target_data_TL = combined_df1[combined_df1['batch'] == "target"].drop(columns=['batch', 'Group']).apply(normalize,axis=1)
    TL_data = combined_df1[combined_df1['batch'] == "TL"].drop(columns=['batch', 'Group']).apply(normalize,axis=1)
    # Target vs raw
    feature_all['batch'] = "Raw"
    combined_df = pd.concat([target[["batch", "Group"] + Raw_feature], feature_all[["batch"] + Raw_feature]],axis=0, ignore_index=True)
    target_data_Raw = combined_df[combined_df['batch'] == "target"].drop(columns=['batch', 'Group']).apply(normalize,axis=1)
    Raw_data = combined_df[combined_df['batch'] == "Raw"].drop(columns=['batch', 'Group']).apply(normalize, axis=1)

    kl_target_TL = compute_kl_between_distributions(TL_data.values,target_data_TL.values)
    kl_target_Raw = compute_kl_between_distributions(Raw_data.values,target_data_Raw.values)
    print(ratio,"kl_target_TL:",kl_target_TL)
    print(ratio, "kl_target_Raw:",kl_target_Raw)

    # Compute JS divergence
    js_target_TL = compute_js_between_distributions(target_data_TL.values,TL_data.values)
    js_target_Raw = compute_js_between_distributions(target_data_Raw.values, Raw_data.values)
    print(ratio, "js_target_TL:", js_target_TL)
    print(ratio, "js_target_Raw:", js_target_Raw)

    # Append the current iteration's results
    iteration_results = pd.DataFrame([{
            "Comparison": "target_vs_TL",
            "KL_Divergence": kl_target_TL,
            "JS_Divergence": js_target_TL,
            "Ratio": ratio
        },
        {
            "Comparison": "target_vs_Raw",
            "KL_Divergence": kl_target_Raw,
            "JS_Divergence": js_target_Raw,
            "Ratio": ratio
        }
    ])
    all_results = pd.concat([all_results, iteration_results], ignore_index=True)
    all_results.to_csv(
        os.path.join(ratio_dir, f"{source_study}_{target_study}_{ratio}_all_divergence_results.csv"),
        index=False
    )
    all_results.to_csv(
        os.path.join(ratio_dir, f"{source_study}_{target_study}_{ratio}_all_divergence_results.csv"),
        index=False
    )
    print(f"Results for ratio {ratio} have been saved.")