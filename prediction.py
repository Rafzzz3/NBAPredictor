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

    projected_pace = (
        float(home_stats["avg_pace"]) + float(away_stats["avg_pace"])
    ) / 2.0
    impact_spread = float(away_absent_impact) - float(home_absent_impact)
    impact_total = float(home_absent_impact) + float(away_absent_impact)

    features = {
        "avg_ortg_home": float(home_stats["avg_ortg"]),
        "avg_drtg_home": float(home_stats["avg_drtg"]),
        "net_rating_home": float(home_stats["net_rating"]),
        "avg_pts_scored_home": float(home_stats["avg_pts_scored"]),
        "avg_pts_lost_home": float(home_stats["avg_pts_lost"]),
        "avg_ortg_away": float(away_stats["avg_ortg"]),
        "avg_drtg_away": float(away_stats["avg_drtg"]),
        "net_rating_away": float(away_stats["net_rating"]),
        "avg_pts_scored_away": float(away_stats["avg_pts_scored"]),
        "avg_pts_lost_away": float(away_stats["avg_pts_lost"]),
        "long_avg_ortg_home": float(home_stats["long_avg_ortg"]),
        "long_avg_drtg_home": float(home_stats["long_avg_drtg"]),
        "long_net_rating_home": float(home_stats["long_net_rating"]),
        "long_avg_pace_home": float(home_stats["long_avg_pace"]),
        "long_avg_ortg_away": float(away_stats["long_avg_ortg"]),
        "long_avg_drtg_away": float(away_stats["long_avg_drtg"]),
        "long_net_rating_away": float(away_stats["long_net_rating"]),
        "long_avg_pace_away": float(away_stats["long_avg_pace"]),
        "projected_pace": float(projected_pace),
        "rest_days_home": int(home_rest),
        "rest_days_away": int(away_rest),
        "b2b_home": 1 if int(home_rest) == 0 else 0,
        "b2b_away": 1 if int(away_rest) == 0 else 0,
        "impact_home": float(home_absent_impact),
        "impact_away": float(away_absent_impact),
        "impact_spread": float(impact_spread),
        "impact_total": float(impact_total),
    }

    x_spread = np.array([[features[col] for col in config["features_spread"]]])
    x_total = np.array([[features[col] for col in config["features_total"]]])

    pred_spread = float(reg_spread.predict(x_spread)[0])
    pred_total = float(reg_total.predict(x_total)[0])

    pts_home = (pred_total + pred_spread) / 2.0
    pts_away = (pred_total - pred_spread) / 2.0

    std_err = config.get("spread_err", config.get("std_spread_err", 11.5))
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
