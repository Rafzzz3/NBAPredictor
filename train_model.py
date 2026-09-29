import os
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error


class LGBTrainer:
    def __init__(self) -> None:
        self.features_spread = [
            "avg_ortg_home",
            "avg_drtg_home",
            "net_rating_home",
            "avg_pts_scored_home",
            "avg_pts_lost_home",
            "avg_ortg_away",
            "avg_drtg_away",
            "net_rating_away",
            "avg_pts_scored_away",
            "avg_pts_lost_away",
            "long_avg_ortg_home",
            "long_avg_drtg_home",
            "long_net_rating_home",
            "long_avg_pace_home",
            "long_avg_ortg_away",
            "long_avg_drtg_away",
            "long_net_rating_away",
            "long_avg_pace_away",
            "projected_pace",
            "rest_days_home",
            "rest_days_away",
            "b2b_home",
            "b2b_away",
            "impact_spread",
        ]
        self.features_total = [
            "avg_ortg_home",
            "avg_drtg_home",
            "net_rating_home",
            "avg_pts_scored_home",
            "avg_pts_lost_home",
            "avg_ortg_away",
            "avg_drtg_away",
            "net_rating_away",
            "avg_pts_scored_away",
            "avg_pts_lost_away",
            "long_avg_ortg_home",
            "long_avg_drtg_home",
            "long_net_rating_home",
            "long_avg_pace_home",
            "long_avg_ortg_away",
            "long_avg_drtg_away",
            "long_net_rating_away",
            "long_avg_pace_away",
            "projected_pace",
            "rest_days_home",
            "rest_days_away",
            "b2b_home",
            "b2b_away",
            "impact_home",
            "impact_away",
        ]

    def optimize(self, X_train, y_train, X_val, y_val, target_name: str) -> dict:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "objective": "regression_l1",
                "metric": "mae",
                "boosting_type": "gbdt",
                "n_estimators": 1000,
                "random_state": 3,
                "n_jobs": -1,
                "verbose": -1,
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.05, log=True
                ),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "subsample": trial.suggest_float("subsample", 0.7, 0.9),
                "subsample_freq": 1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_X=X_val,
                eval_y=y_val,
                callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
            )
            preds = model.predict(X_val)
            return mean_absolute_error(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, 30)
        return study.best_params

    def run_training(self):
        df = pd.read_csv("training_data.csv")
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)
        n = len(df)

        train_idx = int(n * 0.75)
        value_idx = int(n * 0.85)

        train = df.iloc[:train_idx]
        value = df.iloc[train_idx:value_idx]
        test = df.iloc[value_idx:]

        X_train_spread = train[self.features_spread].values
        X_value_spread = value[self.features_spread].values
        X_test_spread = test[self.features_spread].values
        params_spread = self.optimize(
            X_train_spread,
            train["spread"].values,
            X_value_spread,
            value["spread"].values,
            "Spread",
        )
        reg_spread = lgb.LGBMRegressor(
            **params_spread, objective="regression_l1", random_state=3, verbose=-1
        )
        reg_spread.fit(
            X_train_spread,
            train["spread"].values,
            eval_X=X_value_spread,
            eval_y=value["spread"].values,
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )

        X_train_total = train[self.features_total].values
        X_value_total = value[self.features_total].values
        X_test_total = test[self.features_total].values
        params_total = self.optimize(
            X_train_total,
            train["total"].values,
            X_value_total,
            value["total"].values,
            "Total",
        )
        reg_total = lgb.LGBMRegressor(
            **params_total, objective="regression_l1", random_state=3, verbose=-1
        )
        reg_total.fit(
            X_train_total,
            train["total"].values,
            eval_X=X_value_total,
            eval_y=value["total"].values,
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )

        pred_spread = reg_spread.predict(X_test_spread)
        pred_total = reg_total.predict(X_test_total)

        y_spread_test = test["spread"].values
        y_total_test = test["total"].values

        mae_spread = mean_absolute_error(y_spread_test, pred_spread)
        mae_total = mean_absolute_error(y_total_test, pred_total)
        spread_err = float(np.std(y_spread_test - pred_spread))
        win_accuracy = ((pred_spread > 0) == (y_spread_test > 0)).mean() * 100
        print(f"MAE Spreadu:                {mae_spread:.2f} pkt")
        print(f"MAE Sumy punktow (Total):   {mae_total:.2f} pkt")
        print(f"Odchylenie bledu Spreadu:   {spread_err:.2f} pkt")
        print(f"Trafnosc wyboru zwycięzcy: {win_accuracy:.2f}%")

        os.makedirs("models", exist_ok=True)
        joblib.dump(reg_spread, "models/reg_spread.joblib")
        joblib.dump(reg_total, "models/reg_total.joblib")
        joblib.dump(
            {
                "features_spread": self.features_spread,
                "features_total": self.features_total,
                "spread_err": spread_err,
            },
            "models/score_config.pkl",
        )


trainer = LGBTrainer()
trainer.run_training()
