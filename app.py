from datetime import date
import pandas as pd
import streamlit as st
from nba_loader import get_all_teams, get_player_impacts, get_team_stats
from prediction import predict_match

st.set_page_config(
    page_title="NBA Matchday",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Teko:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: radial-gradient(ellipse at 50% -10%, #16243b 0%, #080c14 75%);
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    #MainMenu, header, footer {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 95% !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 18px !important;
        background-color: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
    }

    span[data-baseweb="tag"] {
        border-radius: 12px !important;
        background: rgba(244, 63, 94, 0.2) !important;
        border: 1px solid rgba(244, 63, 94, 0.45) !important;
        color: #fecdd3 !important;
    }

    .team-panel {
        background: linear-gradient(170deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 28px;
        padding: 30px;
        backdrop-filter: blur(20px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.45);
        height: 100%;
    }
    .home-glow {
        border-top: 4px solid #38bdf8;
        box-shadow: 0 12px 35px rgba(56, 189, 248, 0.12);
    }
    .away-glow {
        border-top: 4px solid #fb7185;
        box-shadow: 0 12px 35px rgba(251, 113, 133, 0.12);
    }

    .team-title {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .score-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 32px;
        padding: 40px 20px;
        text-align: center;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin: 30px 0 20px 0;
    }
    .digit-score {
        font-family: 'Teko', sans-serif;
        font-size: 6.8rem;
        line-height: 0.85;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 4px 20px rgba(255, 255, 255, 0.2);
    }

    .stat-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 24px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    }
    .stat-pill-val {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.5px;
    }
    .stat-pill-lbl {
        font-size: 0.82rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #94a3b8;
        margin-top: 5px;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 50%, #c2410c 100%);
        color: white;
        font-weight: 800;
        font-size: 1.35rem;
        border-radius: 22px;
        border: none;
        padding: 1.1rem 2.5rem;
        box-shadow: 0 10px 30px rgba(234, 88, 12, 0.4);
        transition: all 0.25s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 40px rgba(234, 88, 12, 0.6);
        color: white;
    }

    .bar-container {
        display: flex;
        height: 18px;
        border-radius: 30px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.08);
        margin: 10px 0;
        padding: 2px;
    }
    .bar-home {
        background: linear-gradient(90deg, #0284c7, #38bdf8);
        border-radius: 30px 0 0 30px;
        transition: width 0.6s ease;
    }
    .bar-away {
        background: linear-gradient(90deg, #f43f5e, #fda4af);
        border-radius: 0 30px 30px 0;
        transition: width 0.6s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_player_name(raw_name: str) -> str:
    if " (#" in raw_name:
        return raw_name.split(" (#")[0]
    return raw_name


top_l, top_r = st.columns([3.5, 1.2], vertical_alignment="center")

with top_l:
    st.markdown(
        """
        <div style="font-family: 'Teko', sans-serif; font-size: 3.8rem; line-height: 0.95; letter-spacing: 1.5px; background: linear-gradient(90deg, #fb923c, #f43f5e, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            NBA MATCHDAY 
        </div>
        <div style="color: #94a3b8; font-size: 1.05rem; font-weight: 500; margin-top: 4px;">
            Predyktor wyników NBA
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_r:
    sim_date = st.date_input(
        "📅 Data bazy danych:",
        value=date(2024, 4, 14),
        min_value=date(2015, 10, 15),
        max_value=date(2024, 6, 30),
    )
    date_str = sim_date.strftime("%Y-%m-%d")


@st.cache_data(ttl=3600)
def cached_roster(team_id: int, ref_date: str) -> dict[str, float]:
    return get_player_impacts(team_id, end_date=ref_date)


teams_map = get_all_teams()
team_names = sorted(list(teams_map.keys()))

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown('<div class="team-panel home-glow">', unsafe_allow_html=True)
    st.markdown(
        '<div class="team-title" style="color: #38bdf8;">🏠 Drużyna Gospodarzy</div>',
        unsafe_allow_html=True,
    )
    home_team = st.selectbox(
        "Wybierz Gospodarza:",
        team_names,
        index=team_names.index("Boston Celtics")
        if "Boston Celtics" in team_names
        else 0,
        key="home_team_select",
    )
    home_rest = st.slider("Dni odpoczynku:", 0, 5, 2, key="h_rest")
    h_roster = cached_roster(teams_map[home_team], date_str)
    h_out = st.multiselect(
        "Raport kontuzji / nieobecni (OUT):",
        options=list(h_roster.keys()),
        format_func=lambda name: (
            f"{clean_player_name(name)} (-{h_roster[name]:.1f} pkt)"
        ),
        key="h_out_select",
    )
    h_impact = sum(h_roster[p] for p in h_out)
    st.markdown(
        f"<div style='font-size: 0.95rem; color: #94a3b8; margin-top: 10px;'>Ubytek formacji ataku: <b style='color: #f87171; font-size: 1.1rem;'>-{h_impact:.1f} pkt</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="team-panel away-glow">', unsafe_allow_html=True)
    st.markdown(
        '<div class="team-title" style="color: #fb7185;">✈️ Drużyna Gości</div>',
        unsafe_allow_html=True,
    )
    away_team = st.selectbox(
        "Wybierz Gościa:",
        team_names,
        index=team_names.index("Dallas Mavericks")
        if "Dallas Mavericks" in team_names
        else 1,
        key="away_team_select",
    )
    away_rest = st.slider("Dni odpoczynku:", 0, 5, 1, key="a_rest")
    a_roster = cached_roster(teams_map[away_team], date_str)
    a_out = st.multiselect(
        "Raport kontuzji / nieobecni (OUT):",
        options=list(a_roster.keys()),
        format_func=lambda name: (
            f"{clean_player_name(name)} (-{a_roster[name]:.1f} pkt)"
        ),
        key="a_out_select",
    )
    a_impact = sum(a_roster[p] for p in a_out)
    st.markdown(
        f"<div style='font-size: 0.95rem; color: #94a3b8; margin-top: 10px;'>Ubytek formacji ataku: <b style='color: #f87171; font-size: 1.1rem;'>-{a_impact:.1f} pkt</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

if st.button("URUCHOM SYMULACJĘ POJEDYNKU", use_container_width=True):
    if home_team == away_team:
        st.error("Wybierz dwie różne drużyny!")
    else:
        with st.spinner("Przetwarzanie profilu taktycznego i prognozowanie wyniku..."):
            try:
                h_stats = get_team_stats(teams_map[home_team], end_date=date_str)
                a_stats = get_team_stats(teams_map[away_team], end_date=date_str)

                st.session_state["match_result"] = predict_match(
                    home_stats=h_stats,
                    away_stats=a_stats,
                    home_rest=home_rest,
                    away_rest=away_rest,
                    home_absent_impact=h_impact,
                    away_absent_impact=a_impact,
                )
                st.session_state["evaluated_teams"] = (home_team, away_team)
            except Exception as err:
                st.error(f"Błąd silnika symulacji: {err}")
                st.stop()

if "match_result" in st.session_state:
    res = st.session_state["match_result"]
    h_name, a_name = st.session_state.get("evaluated_teams", (home_team, away_team))

    st.markdown(
        f"""
        <div class="score-card">
            <div style="display: flex; justify-content: space-around; align-items: center;">
                <div style="flex: 2;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #38bdf8; margin-bottom: 8px;">{h_name}</div>
                    <div class="digit-score">{round(res["pts_home"])}</div>
                    <div style="color: #64748b; font-size: 0.95rem; font-weight: 600;">{res["pts_home"]:.1f} pkt</div>
                </div>
                <div style="flex: 1; font-family: 'Teko', sans-serif; font-size: 3.2rem; color: #475569;">VS</div>
                <div style="flex: 2;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #fb7185; margin-bottom: 8px;">{a_name}</div>
                    <div class="digit-score">{round(res["pts_away"])}</div>
                    <div style="color: #64748b; font-size: 0.95rem; font-weight: 600;">{res["pts_away"]:.1f} pkt</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prob_h = res["prob_home"]
    prob_a = res["prob_away"]
    st.markdown(
        f"""
        <div style="max-width: 1000px; margin: 0 auto 30px auto;">
            <div style="display: flex; justify-content: space-between; font-weight: 800; font-size: 1.15rem; margin-bottom: 8px;">
                <span style="color: #38bdf8;">{h_name} ({prob_h:.1f}%)</span>
                <span style="color: #fb7185;">({prob_a:.1f}%) {a_name}</span>
            </div>
            <div class="bar-container">
                <div class="bar-home" style="width: {prob_h}%;"></div>
                <div class="bar-away" style="width: {prob_a}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    spread_val = res["spread"]
    spread_str = f"+{spread_val:.1f}" if spread_val > 0 else f"{spread_val:.1f}"

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-pill-val">{spread_str} pkt</div>
                <div class="stat-pill-lbl">Prognozowany Spread (Margines)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-pill-val">{res["total"]:.1f}</div>
                <div class="stat-pill-lbl">Łączna Suma Punktów (O/U)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-pill-val">{res["projected_pace"]:.1f}</div>
                <div class="stat-pill-lbl">Tempo Spotkania (Posiadań / 48m)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
