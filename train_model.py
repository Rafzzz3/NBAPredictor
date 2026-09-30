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
            "impact_total",
        ]

    def _get_monotone_constraints(self, features: list[str], target: str) -> list[int]:
        constraints = []
        for feat in features:
            if target == "Spread":
                if feat == "impact_spread":
                    constraints.append(1)
                else:
                    constraints.append(0)
            elif target == "Total":
                if feat in ["impact_home", "impact_away", "impact_total"]:
                    constraints.append(-1)
                else:
                    constraints.append(0)
            else:
                constraints.append(0)
        return constraints

    def _get_feature_weights(self, features: list[str], target: str) -> list[float]:
        weights = []
        for feat in features:
            if target == "Spread":
                weights.append(3.0 if feat == "impact_spread" else 1.0)
            elif target == "Total":
                weights.append(
                    2.5
                    if feat in ["impact_total", "impact_home", "impact_away"]
                    else 1.0
                )
            else:
                weights.append(1.0)
        return weights

    def optimize(
        self, X_train, y_train, X_val, y_val, features: list[str], target_name: str
    ) -> dict:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        monotone = self._get_monotone_constraints(features, target_name)
        weights = self._get_feature_weights(features, target_name)

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "linear_tree": True,
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "num_leaves": trial.suggest_int("num_leaves", 15, 31),
                "min_child_samples": trial.suggest_int("min_child_samples", 15, 50),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.015, 0.05, log=True
                ),
                "colsample_bytree": 0.8,
                "subsample": trial.suggest_float("subsample", 0.75, 0.95),
                "subsample_freq": 1,
                "monotone_constraints": monotone,
                "feature_weight": weights,
                "n_estimators": 800,
                "random_state": 3,
                "n_jobs": -1,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_X=X_val,
                eval_y=y_val,
                eval_metric="mae",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = model.predict(X_val)
            return mean_absolute_error(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=25)
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
            self.features_spread,
            "Spread",
        )
        params_spread["linear_tree"] = True
        params_spread["colsample_bytree"] = 0.8
        params_spread["subsample_freq"] = 1
        params_spread["monotone_constraints"] = self._get_monotone_constraints(
            self.features_spread, "Spread"
        )
        params_spread["feature_weight"] = self._get_feature_weights(
            self.features_spread, "Spread"
        )
        params_spread["objective"] = "regression"
        params_spread["metric"] = "mae"
        params_spread["n_estimators"] = 800
        params_spread["random_state"] = 3
        params_spread["n_jobs"] = -1
        params_spread["verbose"] = -1

        reg_spread = lgb.LGBMRegressor(**params_spread)
        reg_spread.fit(
            X_train_spread,
            train["spread"].values,
            eval_X=X_value_spread,
            eval_y=value["spread"].values,
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        X_train_total = train[self.features_total].values
        X_value_total = value[self.features_total].values
        X_test_total = test[self.features_total].values

        params_total = self.optimize(
            X_train_total,
            train["total"].values,
            X_value_total,
            value["total"].values,
            self.features_total,
            "Total",
        )
        params_total["linear_tree"] = True
        params_total["colsample_bytree"] = 0.8
        params_total["subsample_freq"] = 1
        params_total["monotone_constraints"] = self._get_monotone_constraints(
            self.features_total, "Total"
        )
        params_total["feature_weight"] = self._get_feature_weights(
            self.features_total, "Total"
        )
        params_total["objective"] = "regression"
        params_total["metric"] = "mae"
        params_total["n_estimators"] = 800
        params_total["random_state"] = 3
        params_total["n_jobs"] = -1
        params_total["verbose"] = -1

        reg_total = lgb.LGBMRegressor(**params_total)
        reg_total.fit(
            X_train_total,
            train["total"].values,
            eval_X=X_value_total,
            eval_y=value["total"].values,
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
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
        print(f"Trafnosc wyboru zwycięzcy:  {win_accuracy:.2f}%")

        spread_imp = dict(zip(self.features_spread, reg_spread.feature_importances_))
        print("\n--- WAŻNOŚĆ CECH DLA SPREADU ---")
        print(f"impact_spread: {spread_imp.get('impact_spread', 0)} podziałów")

        total_imp = dict(zip(self.features_total, reg_total.feature_importances_))
        print("\n--- WAŻNOŚĆ CECH DLA TOTALU ---")
        print(f"impact_home: {total_imp.get('impact_home', 0)} podziałów")
        print(f"impact_away: {total_imp.get('impact_away', 0)} podziałów")
        print(f"impact_total: {total_imp.get('impact_total', 0)} podziałów\n")

        os.makedirs("models", exist_ok=True)
        joblib.dump(reg_spread, "models/reg_spread.joblib")
        joblib.dump(reg_total, "models/reg_total.joblib")
        joblib.dump(
            {
                "features_spread": self.features_spread,
                "features_total": self.features_total,
                "std_spread_err": spread_err,
            },
            "models/score_config.pkl",
        )


trainer = LGBTrainer()
trainer.run_training()
