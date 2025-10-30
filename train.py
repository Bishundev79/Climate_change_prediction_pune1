import numpy as np
import pandas as pd
from src.data_pipeline import DataPipeline
from src.feature_engine import FeatureEngine
from src.ml_models import XGBoostModel, RandomForestModel
from src.dl_models import CNNLSTMModel, TransformerModel, SequenceGenerator
from src.evaluator import Evaluator
from src.config import config
import os

def train_ml_models():
    print("\n" + "="*80)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*80)
    pipeline = DataPipeline()
    train_df, val_df, test_df = pipeline.load_and_prepare()
    feature_eng = FeatureEngine()
    train_feat, feat_cols = feature_eng.create_features(train_df)
    val_feat, _ = feature_eng.create_features(val_df)
    test_feat, _ = feature_eng.create_features(test_df)
    X_train = train_feat[feat_cols].values
    y_train = train_feat[config.TARGET].values
    X_val = val_feat[feat_cols].values
    y_val = val_feat[config.TARGET].values
    X_test = test_feat[feat_cols].values
    y_test = test_feat[config.TARGET].values
    results = {}
    rf = RandomForestModel()
    rf.train(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['Random Forest'] = Evaluator.evaluate(y_test, rf_pred, rf.name)
    rf.save(config.MODEL_DIR / 'random_forest.pkl')
    xgb = XGBoostModel()
    xgb.train(X_train, y_train, X_val, y_val)
    xgb_pred = xgb.predict(X_test)
    results['XGBoost'] = Evaluator.evaluate(y_test, xgb_pred, xgb.name)
    xgb.save(config.MODEL_DIR / 'xgboost.pkl')
    return results

def train_dl_models():
    from sklearn.preprocessing import StandardScaler
    print("\n" + "="*80)
    print("TRAINING DEEP LEARNING MODELS")
    print("="*80)
    pipeline = DataPipeline()
    train_df, val_df, test_df = pipeline.load_and_prepare()
    seq_gen = SequenceGenerator()
    feature_cols = [c for c in train_df.columns if c != config.DATE_COL]
    # ---- SCALE FEATURES & TARGET FOR DEEP LEARNING ----
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train_raw = train_df[feature_cols].values
    X_val_raw   = val_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values
    y_train_raw = train_df[config.TARGET].values.reshape(-1, 1)
    y_val_raw   = val_df[config.TARGET].values.reshape(-1, 1)
    y_test_raw  = test_df[config.TARGET].values.reshape(-1, 1)
    X_train = sc_X.fit_transform(X_train_raw)
    X_val   = sc_X.transform(X_val_raw)
    X_test  = sc_X.transform(X_test_raw)
    y_train = sc_y.fit_transform(y_train_raw).flatten()
    y_val   = sc_y.transform(y_val_raw).flatten()
    y_test  = sc_y.transform(y_test_raw).flatten()
    # ---- CREATE SEQUENCES USING SCALED DATA ----
    X_train_seq, y_train_seq = seq_gen.create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = seq_gen.create_sequences(X_val, y_val)
    X_test_seq, y_test_seq = seq_gen.create_sequences(X_test, y_test)
    print(f"\n📦 Sequence shapes:")
    print(f"   Train: {X_train_seq.shape}")
    print(f"   Val:   {X_val_seq.shape}")
    print(f"   Test:  {X_test_seq.shape}")
    results = {}
    # ---- CNN-LSTM Hybrid ----
    cnn_lstm = CNNLSTMModel()
    cnn_lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    cnn_lstm_pred_scaled = cnn_lstm.predict(X_test_seq)
    cnn_lstm_pred = sc_y.inverse_transform(cnn_lstm_pred_scaled.reshape(-1,1)).flatten()
    y_test_seq_true = sc_y.inverse_transform(y_test_seq.reshape(-1,1)).flatten()
    results['CNN-LSTM'] = Evaluator.evaluate(y_test_seq_true, cnn_lstm_pred, cnn_lstm.name)
    cnn_lstm.model.save(config.MODEL_DIR / 'cnn_lstm.keras')
    # ---- Transformer + Attention ----
    transformer = TransformerModel()
    transformer.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    transformer_pred_scaled = transformer.predict(X_test_seq)
    transformer_pred = sc_y.inverse_transform(transformer_pred_scaled.reshape(-1,1)).flatten()
    results['Transformer'] = Evaluator.evaluate(y_test_seq_true, transformer_pred, transformer.name)
    transformer.model.save(config.MODEL_DIR / 'transformer.keras')
    return results

def save_metrics(all_results):
    os.makedirs("results", exist_ok=True)
    rows = []
    for model, metrics in all_results.items():
        rows.append({
            "Model": model,
            "RMSE": round(metrics['rmse'], 4),
            "MAE": round(metrics['mae'], 4),
            "R2": round(metrics['r2'], 4)
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("RMSE")
    df.to_csv("results/test_metrics.csv", index=False)
    print("\n[INFO] Results saved to results/test_metrics.csv:\n")
    print(df.to_string(index=False))

if __name__ == '__main__':
    print("\n" + "="*80)
    print("CLIMATE CHANGE PREDICTION - PUNE, MAHARASHTRA, INDIA")
    print("="*80)
    ml_results = train_ml_models()
    dl_results = train_dl_models()
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL MODELS")
    print("="*80)
    all_results = {**ml_results, **dl_results}
    for model, metrics in sorted(all_results.items(), key=lambda x: x[1]['rmse']):
        print(f"\n{model}:")
        print(f"   RMSE: {metrics['rmse']:.4f}°C")
        print(f"   R²:   {metrics['r2']:.4f}")
    save_metrics(all_results)
    print("\n✅ Training complete! Models saved to models/")
