import numpy as np
import pandas as pd
from src.data_pipeline import DataPipeline
from src.feature_engine import FeatureEngine
from src.ml_models import XGBoostModel, RandomForestModel

# Try to import DL models, but continue if TensorFlow is not available
try:
    from src.dl_models import CNNLSTMModel, TransformerModel, SequenceGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger_msg = "TensorFlow not available. Only ML models (Random Forest, XGBoost) will be trained."
    print(f"⚠️  {logger_msg}")

from src.evaluator import Evaluator
from src.config import config
from src.logger import logger
import joblib

def train_ml_models():
    logger.info("\n" + "="*80)
    logger.info("TRAINING MACHINE LEARNING MODELS")
    logger.info("="*80)
    
    pipeline = DataPipeline()
    train_df, val_df, test_df = pipeline.load_and_prepare()
    feature_eng = FeatureEngine()
    train_feat, feat_cols = feature_eng.create_features(train_df)
    val_feat, _ = feature_eng.create_features(val_df)
    test_feat, _ = feature_eng.create_features(test_df)
    targets = config.TARGETS if hasattr(config, "TARGETS") else [config.TARGET]
    X_train = train_feat[feat_cols].values
    y_train = train_feat[targets].values
    X_val = val_feat[feat_cols].values
    y_val = val_feat[targets].values
    X_test = test_feat[feat_cols].values
    y_test = test_feat[targets].values
    results = {}
    
    # Train Random Forest
    logger.info("\nTraining Random Forest...")
    rf = RandomForestModel()
    rf.train(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # Evaluate across all targets
    rf_metrics = Evaluator.evaluate(y_test, rf_pred, rf.name)
    results['Random Forest'] = rf_metrics
    rf.save(config.MODEL_DIR / 'random_forest.pkl')
    
    # Save feature importances for Random Forest (average across the 4 Output estimators)
    importances = np.mean([est.feature_importances_ for est in rf.model.estimators_], axis=0)
    feature_importance = pd.DataFrame({
        'feature': feat_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv(config.RESULTS_DIR / 'rf_feature_importance.csv', index=False)
    logger.info("Saved Random Forest feature importances")
    
    # Train XGBoost
    logger.info("\nTraining XGBoost...")
    xgb = XGBoostModel()
    xgb.train(X_train, y_train, X_val, y_val)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = Evaluator.evaluate(y_test, xgb_pred, xgb.name)
    results['XGBoost'] = xgb_metrics
    xgb.save(config.MODEL_DIR / 'xgboost.pkl')
    
    # Save feature importances for XGBoost
    xgb_importances = np.mean([est.feature_importances_ for est in xgb.model.estimators_], axis=0)
    feature_importance = pd.DataFrame({
        'feature': feat_cols,
        'importance': xgb_importances
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv(config.RESULTS_DIR / 'xgb_feature_importance.csv', index=False)
    logger.info("Saved XGBoost feature importances")
    
    return results, y_test, {'rf': rf_pred, 'xgb': xgb_pred}

def train_dl_models():
    """Train deep learning models (requires TensorFlow)"""
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available. Skipping deep learning models.")
        return {}, None, {}
    
    from sklearn.preprocessing import StandardScaler
    logger.info("\n" + "="*80)
    logger.info("TRAINING DEEP LEARNING MODELS")
    logger.info("="*80)
    
    pipeline = DataPipeline()
    train_df, val_df, test_df = pipeline.load_and_prepare()
    seq_gen = SequenceGenerator()
    feature_cols = [c for c in train_df.columns if c != config.DATE_COL]
    
    # Scale features & target for deep learning
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    targets = config.TARGETS if hasattr(config, "TARGETS") else [config.TARGET]
    X_train_raw = train_df[feature_cols].values
    X_val_raw   = val_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values
    y_train_raw = train_df[targets].values
    y_val_raw   = val_df[targets].values
    y_test_raw  = test_df[targets].values
    
    X_train = sc_X.fit_transform(X_train_raw)
    X_val   = sc_X.transform(X_val_raw)
    X_test  = sc_X.transform(X_test_raw)
    y_train = sc_y.fit_transform(y_train_raw)
    y_val   = sc_y.transform(y_val_raw)
    y_test  = sc_y.transform(y_test_raw)
    
    # Create sequences using scaled data
    X_train_seq, y_train_seq = seq_gen.create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = seq_gen.create_sequences(X_val, y_val)
    X_test_seq, y_test_seq = seq_gen.create_sequences(X_test, y_test)
    
    logger.info(f"\n📦 Sequence shapes:")
    logger.info(f"   Train: {X_train_seq.shape}")
    logger.info(f"   Val:   {X_val_seq.shape}")
    logger.info(f"   Test:  {X_test_seq.shape}")
    
    results = {}
    
    # Train CNN-LSTM Hybrid
    logger.info("\nTraining CNN-LSTM Hybrid...")
    cnn_lstm = CNNLSTMModel()
    cnn_lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    
    num_targets = len(targets)
    cnn_lstm_pred_scaled = cnn_lstm.predict(X_test_seq)
    cnn_lstm_pred_reshaped = cnn_lstm_pred_scaled.reshape(-1, num_targets)
    cnn_lstm_pred = sc_y.inverse_transform(cnn_lstm_pred_reshaped)
    
    y_test_seq_true = sc_y.inverse_transform(y_test_seq.reshape(-1, num_targets))
    results['CNN-LSTM'] = Evaluator.evaluate(y_test_seq_true, cnn_lstm_pred, cnn_lstm.name)
    cnn_lstm.model.save(config.MODEL_DIR / 'cnn_lstm.keras')
    
    # Train Transformer + Attention
    logger.info("\nTraining Transformer...")
    transformer = TransformerModel()
    transformer.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    
    transformer_pred_scaled = transformer.predict(X_test_seq)
    transformer_pred_reshaped = transformer_pred_scaled.reshape(-1, num_targets)
    transformer_pred = sc_y.inverse_transform(transformer_pred_reshaped)
    
    results['Transformer'] = Evaluator.evaluate(y_test_seq_true, transformer_pred, transformer.name)
    transformer.model.save(config.MODEL_DIR / 'transformer.keras')

    # Save scalers so the Forecast page can reuse them without re-fitting
    joblib.dump(sc_X, config.MODEL_DIR / 'scaler_X.pkl')
    joblib.dump(sc_y, config.MODEL_DIR / 'scaler_y.pkl')
    logger.info("Saved DL scalers (scaler_X.pkl, scaler_y.pkl)")

    return results, y_test_seq_true, {'cnn_lstm': cnn_lstm_pred, 'transformer': transformer_pred}

def save_metrics(all_results):
    """Save model metrics to CSV file"""
    config.RESULTS_DIR.mkdir(exist_ok=True)
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
    df.to_csv(config.RESULTS_DIR / "test_metrics.csv", index=False)
    logger.info("\n[INFO] Results saved to results/test_metrics.csv:\n")
    logger.info("\n" + df.to_string(index=False))


def calculate_ensemble(y_true, predictions):
    """Calculate ensemble prediction using weighted average"""
    # Simple average ensemble
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    return ensemble_pred


if __name__ == '__main__':
    logger.info("\n" + "="*80)
    logger.info("CLIMATE CHANGE PREDICTION - PUNE, MAHARASHTRA, INDIA")
    logger.info("="*80)
    
    if not TENSORFLOW_AVAILABLE:
        logger.warning("\n⚠️  TensorFlow is not available. Only ML models will be trained.")
        logger.info("To install TensorFlow, use: pip install tensorflow")
    
    try:
        # Train ML models
        ml_results, ml_y_test, ml_predictions = train_ml_models()
        
        # Train DL models (if TensorFlow is available)
        dl_results, dl_y_test, dl_predictions = train_dl_models()
        
        all_results = {**ml_results}
        
        if TENSORFLOW_AVAILABLE and dl_results:
            all_results.update(dl_results)
        
        logger.info("\n" + "="*80)
        logger.info("FINAL RESULTS - ALL MODELS")
        logger.info("="*80)
        
        for model, metrics in sorted(all_results.items(), key=lambda x: x[1]['rmse']):
            logger.info(f"\n{model}:")
            logger.info(f"   RMSE: {metrics['rmse']:.4f}°C")
            logger.info(f"   R²:   {metrics['r2']:.4f}")
        
        save_metrics(all_results)
        logger.info("\n✅ Training complete! Models saved to models/")
        if not TENSORFLOW_AVAILABLE:
            logger.info("Note: Only ML models were trained (TensorFlow not available)")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        logger.exception("Full traceback:")
        raise
