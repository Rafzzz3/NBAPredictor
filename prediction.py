import os
import joblib
import numpy as np
from scipy.stats import norm


def load_prediction_models(
    models_dir: str = "models",
) -> tuple[dict, any, any]:
    path_spread = os.path.join(models_dir, "reg_spread.joblib")
    path_total = os.path.join(models_dir, "reg_total.joblib")
    path_cfg = os.path.join(models_dir, "score_config.pkl")

    reg_spread = joblib.load(path_spread)
    reg_total = joblib.load(path_total)
    config = joblib.load(path_cfg)

    return config, reg_spread, reg_total


def predict_match(
    home_stats: dict[str, float],
    away_stats: dict[str, float],
    home_rest: int = 2,
    away_rest: int = 2,
    home_absent_impact: float = 0.0,
    away_absent_impact: float = 0.0,
    models_dir: str = "models",
) -> dict[str, any]:
    config, reg_spread, reg_total = load_prediction_models(models_dir)

    projected_pace = (home_stats["avg_pace"] + away_stats["avg_pace"]) / 2.0
    impact_spread = away_absent_impact - home_absent_impact

    features = {
        "avg_ortg_home": home_stats["avg_ortg"],
        "avg_drtg_home": home_stats["avg_drtg"],
        "net_rating_home": home_stats["net_rating"],
        "avg_pts_scored_home": home_stats["avg_pts_scored"],
        "avg_pts_lost_home": home_stats["avg_pts_lost"],
        "avg_ortg_away": away_stats["avg_ortg"],
        "avg_drtg_away": away_stats["avg_drtg"],
        "net_rating_away": away_stats["net_rating"],
        "avg_pts_scored_away": away_stats["avg_pts_scored"],
        "avg_pts_lost_away": away_stats["avg_pts_lost"],
        "long_avg_ortg_home": home_stats["long_avg_ortg"],
        "long_avg_drtg_home": home_stats["long_avg_drtg"],
        "long_net_rating_home": home_stats["long_net_rating"],
        "long_avg_pace_home": home_stats["long_avg_pace"],
        "long_avg_ortg_away": away_stats["long_avg_ortg"],
        "long_avg_drtg_away": away_stats["long_avg_drtg"],
        "long_net_rating_away": away_stats["long_net_rating"],
        "long_avg_pace_away": away_stats["long_avg_pace"],
        "projected_pace": projected_pace,
        "rest_days_home": home_rest,
        "rest_days_away": away_rest,
        "b2b_home": 1 if home_rest <= 1 else 0,
        "b2b_away": 1 if away_rest <= 1 else 0,
        "impact_home": home_absent_impact,
        "impact_away": away_absent_impact,
        "impact_spread": impact_spread,
    }

    x_spread = np.array([[features[col] for col in config["features_spread"]]])
    x_total = np.array([[features[col] for col in config["features_total"]]])

    pred_spread = float(reg_spread.predict(x_spread)[0])
    pred_total = float(reg_total.predict(x_total)[0])

    pts_home = (pred_total + pred_spread) / 2.0
    pts_away = (pred_total - pred_spread) / 2.0

    std_err = config["std_spread_err"]
    prob_home = float(1.0 - norm.cdf(0, loc=pred_spread, scale=std_err))
    prob_away = 1.0 - prob_home

    return {
        "exact_score": f"{round(pts_home)} : {round(pts_away)}",
        "pts_home": round(pts_home, 1),
        "pts_away": round(pts_away, 1),
        "spread": round(pred_spread, 1),
        "total": round(pred_total, 1),
        "prob_home": round(prob_home * 100, 1),
        "prob_away": round(prob_away * 100, 1),
        "projected_pace": round(projected_pace, 1),
    }
