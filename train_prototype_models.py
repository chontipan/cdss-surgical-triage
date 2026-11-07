import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

def train_and_save():
    models_dir = Path(r'D:\Appendix\cdss_prototype\models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    reader = pd.io.stata.StataReader(r'D:\Appendix\dataset_review_443\imputed_full_dataset_443.dta')
    vl = reader.value_labels()['timehr']
    df = pd.read_stata(r'D:\Appendix\dataset_review_443\imputed_full_dataset_443.dta', convert_categoricals=False)

    def decode_time(val):
        s = vl.get(int(val), str(val)).strip()
        if 'min' in s: return float(s.replace('min', '').strip()) / 60.0
        return float(s)

    df['time_hours'] = df['timehr'].apply(decode_time)
    df['log_time'] = np.log1p(df['time_hours'])
    df['rr_clean'] = df['rr'].clip(upper=60)
    df['hr_clean'] = df['hr'].clip(lower=40)
    df['shock_index'] = df['hr_clean'] / df['sbp'].replace(0, np.nan)
    df['pulse_pressure'] = df['sbp'] - df['dbp']
    df['fever'] = (df['bt'] >= 38.0).astype(float)
    df['delay_24h'] = (df['time_hours'] >= 24.0).astype(float)
    df['delay_48h'] = (df['time_hours'] >= 48.0).astype(float)
    
    # Tier 2 lab features
    df['nlr'] = df['pmn'] / (100.0 - df['pmn']).clip(lower=1.0)
    df['wbc_marked'] = (df['wbc'] >= 15000).astype(float)
    df['wbc_pmn_interaction'] = (df['wbc'] / 10000.0) * (df['pmn'] / 100.0)
    
    y = df['dx'].astype(int).values
    
    # Tier 1 feature set
    t1_cols = ['age', 'gender', 'bt', 'sbp', 'dbp', 'hr_clean', 'rr_clean', 
               'time_hours', 'log_time', 'shock_index', 'pulse_pressure', 'fever', 'delay_24h', 'delay_48h']
    
    # Tier 2 feature set
    t2_cols = t1_cols + ['wbc', 'pmn', 'nlr', 'wbc_marked', 'wbc_pmn_interaction']
    
    X1 = df[t1_cols]
    X2 = df[t2_cols]
    
    # Train Tier 1 Model (Balanced Logistic Regression)
    pipe1 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=20261106))
    ])
    pipe1.fit(X1, y)
    joblib.dump({'model': pipe1, 'features': t1_cols}, models_dir / 'tier1_pipeline.joblib')
    print("Tier 1 pipeline saved. Features:", len(t1_cols))
    
    # Train Tier 2 LR Model
    pipe2_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.8, class_weight='balanced', max_iter=2000, random_state=20261106))
    ])
    pipe2_lr.fit(X2, y)
    joblib.dump({'model': pipe2_lr, 'features': t2_cols}, models_dir / 'tier2_lr_pipeline.joblib')
    print("Tier 2 LR pipeline saved. Features:", len(t2_cols))
    
    # Train Tier 2 XGBoost Model
    pipe2_xgb = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.05, 
            scale_pos_weight=(len(y) - sum(y)) / sum(y),
            random_state=20261106, 
            eval_metric='logloss'
        ))
    ])
    pipe2_xgb.fit(X2, y)
    joblib.dump({'model': pipe2_xgb, 'features': t2_cols}, models_dir / 'tier2_xgb_pipeline.joblib')
    print("Tier 2 XGB pipeline saved.")
    
    # Save feature names and training baseline summary
    summary = {
        'n_train': len(df),
        'ruptured_n': int(y.sum()),
        'non_ruptured_n': int(len(y) - y.sum()),
        't1_features': t1_cols,
        't2_features': t2_cols
    }
    joblib.dump(summary, models_dir / 'cohort_summary.joblib')
    print("All prototype models successfully trained and serialized!")

if __name__ == '__main__':
    train_and_save()
