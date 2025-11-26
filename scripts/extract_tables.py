#!/usr/bin/env python3
"""
Extract tables from tables/all_results.json.

Generates:
1. table2_architecture_comparison.csv
2. table3_generalization.csv
3. table4a_ablation_loss.csv
4. table4b_ablation_dim.csv
5. table4c_knn.csv
"""

import json
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
TABLES_DIR = PROJECT_DIR / "tables"
RESULTS_JSON = TABLES_DIR / "all_results.json"
KNN_RESULTS_JSON = TABLES_DIR / "knn_results.json"

def load_results() -> pd.DataFrame:
    """Load results from all_results.json and knn_results.json."""
    df_all = pd.DataFrame()
    
    # Load main results
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, 'r') as f:
                data = json.load(f)
            df_all = pd.DataFrame(data)
            print(f"✅ Loaded {len(df_all)} rows from {RESULTS_JSON}")
        except Exception as e:
            print(f"❌ Error loading {RESULTS_JSON}: {e}")
    else:
        print(f"⚠️ {RESULTS_JSON} not found!")

    # Load K-NN results and merge
    if KNN_RESULTS_JSON.exists():
        try:
            with open(KNN_RESULTS_JSON, 'r') as f:
                knn_data = json.load(f)
            df_knn = pd.DataFrame(knn_data)
            
            # Add missing columns to match main dataframe structure
            # Assuming K-NN results are for mmcbnu/resnet50_cbam/triplet_center/256
            if 'dataset' not in df_knn.columns: df_knn['dataset'] = 'mmcbnu'
            if 'model' not in df_knn.columns: df_knn['model'] = 'resnet50_cbam'
            if 'loss' not in df_knn.columns: df_knn['loss'] = 'triplet_center'
            if 'embedding_dim' not in df_knn.columns: df_knn['embedding_dim'] = '256'
            
            print(f"✅ Loaded {len(df_knn)} rows from {KNN_RESULTS_JSON}")
            
            # Append to main dataframe
            df_all = pd.concat([df_all, df_knn], ignore_index=True)
            
        except Exception as e:
            print(f"❌ Error loading {KNN_RESULTS_JSON}: {e}")
            
    return df_all

def save_table(df: pd.DataFrame, filename: str, title: str):
    if df.empty:
        print(f"⚠️ {title}: No data found matching criteria.")
        return
        
    # Sort by OSCR descending if present
    if 'oscr' in df.columns:
        df = df.sort_values('oscr', ascending=False)
        
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    print(f"📊 {title} saved to {path}")
    print(df.to_string(index=False))
    print("-" * 80)

def main():
    df = load_results()
    if df.empty:
        return

    # Ensure columns exist
    required_cols = ['dataset', 'model', 'loss', 'embedding_dim', 'k', 'oscr']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            return

    # Normalize columns
    df['k'] = pd.to_numeric(df['k'], errors='coerce').fillna(1).astype(int)
    df['embedding_dim'] = df['embedding_dim'].astype(str)

    # =========================================================================
    # Table 2: Architecture Comparison
    # Filter: dataset=mmcbnu, k=1. Group by model, take best OSCR.
    # =========================================================================
    print("\nGenerating Table 2: Architecture Comparison...")
    t2 = df[
        (df['dataset'] == 'mmcbnu') & 
        (df['k'] == 1)
    ].copy()
    
    if not t2.empty:
        # Group by model and take best OSCR
        t2_best = t2.loc[t2.groupby('model')['oscr'].idxmax()]
        cols_t2 = ['model', 'oscr', 'auroc', 'eer', 'rank1', 'accuracy', 'query_time_per_sample_ms']
        # Filter columns that actually exist
        cols_t2 = [c for c in cols_t2 if c in t2_best.columns]
        save_table(t2_best[cols_t2], "table2_architecture_comparison.csv", "Table 2 (Architecture)")
    else:
        print("⚠️ Table 2: No data found.")

    # =========================================================================
    # Table 3: Generalization
    # Filter: model=resnet50_cbam (or resnet50_cbam_center), k=1. Group by dataset, take best.
    # Note: In all_results.json, model is 'resnet50_cbam'.
    # =========================================================================
    print("\nGenerating Table 3: Generalization...")
    # Check which model name is used for the best model
    best_model_name = 'resnet50_cbam' 
    # Or check if 'resnet50_cbam_center' exists
    if 'resnet50_cbam_center' in df['model'].unique():
        best_model_name = 'resnet50_cbam_center'
    
    t3 = df[
        (df['model'] == best_model_name) & 
        (df['k'] == 1)
    ].copy()
    
    if not t3.empty:
        t3_best = t3.loc[t3.groupby('dataset')['oscr'].idxmax()]
        cols_t3 = ['dataset', 'oscr', 'auroc', 'eer', 'rank1', 'accuracy']
        cols_t3 = [c for c in cols_t3 if c in t3_best.columns]
        save_table(t3_best[cols_t3], "table3_generalization.csv", "Table 3 (Generalization)")
    else:
        print(f"⚠️ Table 3: No data found for model {best_model_name}.")

    # =========================================================================
    # Table 4a: Ablation - Loss
    # Filter: dataset=mmcbnu, model=resnet50_cbam, dim=256, k=1
    # =========================================================================
    print("\nGenerating Table 4a: Ablation (Loss)...")
    t4a = df[
        (df['dataset'] == 'mmcbnu') & 
        (df['model'] == best_model_name) & 
        (df['embedding_dim'] == '256') & 
        (df['k'] == 1)
    ].copy()
    
    if not t4a.empty:
        t4a = t4a.loc[t4a.groupby('loss')['oscr'].idxmax()]
        cols_t4a = ['loss', 'oscr', 'auroc', 'eer', 'rank1']
        cols_t4a = [c for c in cols_t4a if c in t4a.columns]
        save_table(t4a[cols_t4a], "table4a_ablation_loss.csv", "Table 4a (Loss)")
    else:
        print("⚠️ Table 4a: No data found.")

    # =========================================================================
    # Table 4b: Ablation - Dimension
    # Filter: dataset=mmcbnu, model=resnet50_cbam, loss=triplet_center, k=1
    # =========================================================================
    print("\nGenerating Table 4b: Ablation (Dimension)...")
    t4b = df[
        (df['dataset'] == 'mmcbnu') & 
        (df['model'] == best_model_name) & 
        (df['loss'] == 'triplet_center') & 
        (df['k'] == 1)
    ].copy()
    
    if not t4b.empty:
        t4b = t4b.loc[t4b.groupby('embedding_dim')['oscr'].idxmax()]
        # Sort by dim numerically
        t4b['dim_int'] = pd.to_numeric(t4b['embedding_dim'], errors='coerce')
        t4b = t4b.sort_values('dim_int')
        cols_t4b = ['embedding_dim', 'oscr', 'auroc', 'eer', 'rank1']
        cols_t4b = [c for c in cols_t4b if c in t4b.columns]
        save_table(t4b[cols_t4b], "table4b_ablation_dim.csv", "Table 4b (Dimension)")
    else:
        print("⚠️ Table 4b: No data found.")

    # =========================================================================
    # Table 4c: K-NN
    # Filter: dataset=mmcbnu, model=resnet50_cbam, loss=triplet_center, dim=256
    # =========================================================================
    print("\nGenerating Table 4c: K-NN...")
    t4c = df[
        (df['dataset'] == 'mmcbnu') & 
        (df['model'] == best_model_name) & 
        (df['loss'] == 'triplet_center') & 
        (df['embedding_dim'] == '256')
    ].copy()
    
    if not t4c.empty:
        t4c = t4c.loc[t4c.groupby('k')['oscr'].idxmax()]
        t4c = t4c.sort_values('k')
        cols_t4c = ['k', 'oscr', 'auroc', 'eer', 'rank1']
        cols_t4c = [c for c in cols_t4c if c in t4c.columns]
        save_table(t4c[cols_t4c], "table4c_knn.csv", "Table 4c (K-NN)")
    else:
        print("⚠️ Table 4c: No data found.")

if __name__ == "__main__":
    main()
