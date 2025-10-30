import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n📊 {model_name} Results:")
        print(f"   RMSE: {rmse:.4f}°C")
        print(f"   MAE:  {mae:.4f}°C")
        print(f"   R²:   {r2:.4f}")
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
