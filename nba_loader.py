import pandas as pd
from nba_api.stats.static import teams


def get_all_teams() -> dict[str, int]:
    return {t["full_name"]: t["id"] for t in teams.get_teams()}


def get_team_stats(team_id: int, end_date: str = "2024-04-01") -> dict[str, float]:
    df = pd.read_csv("csv/game.csv")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"] <= pd.to_datetime(end_date)]

    home = df[df["team_id_home"] == team_id].copy()
    away = df[df["team_id_away"] == team_id].copy()

    home["pts_scored"] = home["pts_home"]
    home["pts_conceded"] = home["pts_away"]
    home["poss"] = (
        home["fga_home"]
        + 0.44 * home["fta_home"]
        - home["oreb_home"]
        + home["tov_home"]
    )

    away["pts_scored"] = away["pts_away"]
    away["pts_conceded"] = away["pts_home"]
    away["poss"] = (
        away["fga_away"]
        + 0.44 * away["fta_away"]
        - away["oreb_away"]
        + away["tov_away"]
    )

    combined = pd.concat([home, away], ignore_index=True).sort_values("game_date")
    if len(combined) < 3:
        raise ValueError(f"Zbyt mało meczów dla drużyny ID {team_id} przed {end_date}.")

    recent_7 = combined.tail(7)
    poss_7 = max(float(recent_7["poss"].mean()), 1.0)
    pts_s_7 = float(recent_7["pts_scored"].mean())
    pts_l_7 = float(recent_7["pts_conceded"].mean())
    ortg_7 = round((pts_s_7 * 100.0) / poss_7, 1)
    drtg_7 = round((pts_l_7 * 100.0) / poss_7, 1)

    recent_20 = combined.tail(20)
    poss_20 = max(float(recent_20["poss"].mean()), 1.0)
    pts_s_20 = float(recent_20["pts_scored"].mean())
    pts_l_20 = float(recent_20["pts_conceded"].mean())
    ortg_20 = round((pts_s_20 * 100.0) / poss_20, 1)
    drtg_20 = round((pts_l_20 * 100.0) / poss_20, 1)

    return {
        "avg_ortg": ortg_7,
        "avg_drtg": drtg_7,
        "net_rating": round(ortg_7 - drtg_7, 1),
        "avg_pts_scored": round(pts_s_7, 1),
        "avg_pts_lost": round(pts_l_7, 1),
        "avg_pace": round(poss_7, 1),
        "long_avg_ortg": ortg_20,
        "long_avg_drtg": drtg_20,
        "long_net_rating": round(ortg_20 - drtg_20, 1),
        "long_avg_pace": round(poss_20, 1),
    }


def get_player_impacts(team_id: int, end_date: str = "2024-04-01") -> dict[str, float]:
    df = pd.read_csv(
        "csv/PlayerStatistics.csv",
        usecols=[
            "firstName",
            "lastName",
            "personId",
            "playerteamId",
            "numMinutes",
            "points",
            "gameDate",
        ],
        low_memory=False,
    )
    df["gameDate"] = pd.to_datetime(df["gameDate"])
    target_date = pd.to_datetime(end_date)
    start_window = target_date - pd.Timedelta(days=365)

    df_window = df[
        (df["gameDate"] <= target_date) & (df["gameDate"] >= start_window)
    ].copy()
    if df_window.empty:
        return {}

    last_game_per_player = df_window.sort_values("gameDate").groupby("personId").last()
    current_team_players = last_game_per_player[
        last_game_per_player["playerteamId"] == team_id
    ].index

    df_team = df_window[
        (df_window["playerteamId"] == team_id)
        & (df_window["personId"].isin(current_team_players))
    ].copy()

    if df_team.empty:
        return {}

    df_team["playerName"] = (
        df_team["firstName"].fillna("")
        + " "
        + df_team["lastName"].fillna("")
        + " (#"
        + df_team["personId"].astype(str)
        + ")"
    ).str.strip()

    df_team["min"] = pd.to_numeric(
        df_team["numMinutes"].astype(str).str.split(":").str[0],
        errors="coerce",
    ).fillna(0.0)
    df_team["pts"] = pd.to_numeric(df_team["points"], errors="coerce").fillna(0.0)

    df_team = df_team.sort_values("gameDate")

    recent_matches = df_team.groupby("playerName").tail(7)
    stats = recent_matches.groupby("playerName").agg(
        pts=("pts", "mean"), min=("min", "mean"), games=("pts", "count")
    )
    stats = stats[stats["games"] >= 3]

    stats["wage"] = (stats["pts"] * (stats["min"] / 48.0)).round(1)
    stats = stats[stats["wage"] >= 0.5]

    return stats["wage"].sort_values(ascending=False).to_dict()
