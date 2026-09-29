import numpy as np
import pandas as pd


def find_player_impact(df_inactive: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(
        "csv/PlayerStatistics.csv",
        usecols=[
            "gameId",
            "personId",
            "playerteamId",
            "numMinutes",
            "points",
            "gameDate",
        ],
        low_memory=False,
    )
    df = df.rename(
        columns={
            "gameId": "game_id",
            "personId": "player_id",
            "playerteamId": "team_id",
            "numMinutes": "min",
            "points": "pts",
            "gameDate": "game_date",
        }
    )
    df = df.dropna(subset=["game_id", "player_id", "team_id"]).copy()
    df_inactive = df_inactive.dropna(subset=["game_id", "player_id"]).copy()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"] >= "2015-09-01"].copy()
    df["game_id"] = df["game_id"].astype("int64")
    df["player_id"] = df["player_id"].astype("int64")
    df["team_id"] = df["team_id"].astype("int64")

    df_inactive["game_id"] = df_inactive["game_id"].astype("int64")
    df_inactive["player_id"] = df_inactive["player_id"].astype("int64")

    df["min"] = pd.to_numeric(df["min"], errors="coerce")
    df["min"] = df["min"].fillna(0)
    df["pts"] = pd.to_numeric(df["pts"], errors="coerce")
    df["pts"] = df["pts"].fillna(0)

    df = df.sort_values(by=["player_id", "game_id"]).reset_index(drop=True)
    group = df.groupby("player_id")[["pts", "min"]]
    avg = group.shift(1).rolling(7, min_periods=3).mean()
    df["wage"] = avg["pts"] * (avg["min"] / 48.0)

    merged = df_inactive.merge(
        df[["game_id", "player_id", "wage"]], on=["game_id", "player_id"]
    )

    result = merged.groupby(["game_id", "team_id"], as_index=False)["wage"].sum()
    result.columns = ["game_id", "team_id", "impact"]

    return result


def process_data(
    game: str = "csv/game.csv",
    other_stats: str = "csv/other_stats.csv",
    inactive: str = "csv/inactive_players.csv",
    output: str = "training_data.csv",
) -> pd.DataFrame:
    df = pd.read_csv(game)
    df = df[df["season_type"].isin(["Regular Season", "Playoffs"])].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"] >= "2015-10-01"].copy()
    df = df.sort_values(by=["game_date"]).reset_index(drop=True)
    df = df.dropna(subset=["pts_home", "pts_away", "min", "fga_home", "fga_away"])
    # Liczenie średniej ilośći posiadań zgodnie z ogoólnodostępnym wzorem
    possesion_home = (
        df["fga_home"] + 0.44 * df["fta_home"] - df["oreb_home"] + df["tov_home"]
    )
    possesion_away = (
        df["fga_away"] + 0.44 * df["fta_away"] - df["oreb_away"] + df["tov_away"]
    )
    game_possesion = (possesion_away + possesion_home) / 2.0

    df["pace"] = (game_possesion / df["min"]) * 240.0
    df["ortg_home"] = (df["pts_home"] * 100.0) / possesion_home.replace(0, np.nan)
    df["ortg_away"] = (df["pts_away"] * 100.0) / possesion_away.replace(0, np.nan)

    df_inactive = pd.read_csv(inactive)
    df_impact = find_player_impact(df_inactive)

    df = df.merge(
        df_impact.rename(
            columns={
                "team_id": "team_id_home",
                "impact": "impact_home",
            }
        ),
        on=["game_id", "team_id_home"],
        how="left",
    ).merge(
        df_impact.rename(
            columns={
                "team_id": "team_id_away",
                "impact": "impact_away",
            }
        ),
        on=["game_id", "team_id_away"],
        how="left",
    )

    df["impact_home"] = df["impact_home"].fillna(0.0)
    df["impact_away"] = df["impact_away"].fillna(0.0)
    df["impact_spread"] = df["impact_away"] - df["impact_home"]

    home = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "game_date": df["game_date"],
            "team_id": df["team_id_home"],
            "is_home": 1,
            "pts_scored": df["pts_home"],
            "pts_lost": df["pts_away"],
            "ortg": df["ortg_home"],
            "drtg": df["ortg_away"],
            "pace": df["pace"],
        }
    )
    away = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "game_date": df["game_date"],
            "team_id": df["team_id_away"],
            "is_home": 0,
            "pts_scored": df["pts_away"],
            "pts_lost": df["pts_home"],
            "ortg": df["ortg_away"],
            "drtg": df["ortg_home"],
            "pace": df["pace"],
        }
    )
    team_stats = pd.concat([home, away], ignore_index=True)
    team_stats = team_stats.sort_values(by=["team_id", "game_date"]).reset_index(
        drop=True
    )
    group = team_stats.groupby("team_id")
    cols = [
        "ortg",
        "drtg",
        "pts_scored",
        "pts_lost",
        "pace",
    ]
    for col in cols:
        team_stats[f"avg_{col}"] = (
            ## shift zapobiega data lekage
            group[col].shift(1).rolling(7, min_periods=3).mean()
        )
    team_stats["net_rating"] = team_stats["avg_ortg"] - team_stats["avg_drtg"]

    for col in ["ortg", "drtg", "pace"]:
        team_stats[f"long_avg_{col}"] = (
            group[col].shift(1).rolling(20, min_periods=7).mean()
        )

    team_stats["long_net_rating"] = (
        team_stats["long_avg_ortg"] - team_stats["long_avg_drtg"]
    )
    ## shift znowu zapobiega data leckage
    days_diff = (team_stats["game_date"] - group["game_date"].shift(1)).dt.days
    team_stats["rest_days"] = days_diff.fillna(3).clip(lower=0, upper=7)
    team_stats["b2b"] = (team_stats["rest_days"] <= 1).astype(int)

    features_cols = (
        [f"avg_{col}" for col in cols]
        + ["rest_days", "net_rating"]
        + [f"long_avg_{col}" for col in ["ortg", "drtg", "pace"]]
        + ["long_net_rating", "rest_days", "b2b"]
    )

    sub_home = team_stats.loc[team_stats["is_home"] == 1]
    features_home = sub_home[features_cols].add_suffix("_home")
    features_home["game_id"] = sub_home["game_id"].values

    sub_away = team_stats.loc[team_stats["is_home"] == 0]
    features_away = sub_away[features_cols].add_suffix("_away")
    features_away["game_id"] = sub_away["game_id"].values
    result = df[
        [
            "game_id",
            "game_date",
            "team_id_home",
            "team_id_away",
            "pts_home",
            "pts_away",
            "impact_home",
            "impact_away",
            "impact_spread",
        ]
    ].copy()

    result = result.merge(features_home, on="game_id", how="inner")
    result = result.merge(features_away, on="game_id", how="inner")
    result = result.dropna().reset_index(drop=True)

    result["spread"] = result["pts_home"] - result["pts_away"]
    result["total"] = result["pts_home"] + result["pts_away"]
    result["projected_pace"] = (result["avg_pace_home"] + result["avg_pace_away"]) / 2.0

    result.to_csv(output, index=False)

    return result


result = process_data()
print(result.head())
