import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import time

st.set_page_config(page_title="Arsenal Pro AI - Real-Time Tactics", layout="wide")

# ---------------------------------------------------------
# 1. DATA PARSING & SESSION STATE INITS
# ---------------------------------------------------------
@st.cache_data
def load_data():
    url_home = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
    url_away = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
    
    df_h = pd.read_csv(url_home, skiprows=2)
    df_a = pd.read_csv(url_away, skiprows=2)
    
    def clean_metrica(df, prefix):
        new_cols = []
        cols = df.columns
        for i, col in enumerate(cols):
            if i == 0: new_cols.append('Period')
            elif i == 1: new_cols.append('Frame')
            elif i == 2: new_cols.append('Time')
            elif 'Ball' in col or (i >= len(cols)-2):
                new_cols.append('Ball_x' if i % 2 != 0 else 'Ball_y')
            else:
                p_id = cols[i if i % 2 != 0 else i-1].replace('Player', '')
                axis = 'x' if i % 2 != 0 else 'y'
                new_cols.append(f'{prefix}_{p_id}_{axis}')
        df.columns = new_cols
        return df

    return clean_metrica(df_h, 'Home'), clean_metrica(df_a, 'Away')

df_home, df_away = load_data()
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Initialize playback session states
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 100
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

# ---------------------------------------------------------
# 2. PHYSICS & PASS INTERCEPTION ENGINES
# ---------------------------------------------------------
def compute_kinematic_pitch_control(home_pos, away_pos, grid_res=(30, 20)):
    reaction_time = 0.7; v_max = 7.0
    x = np.linspace(0, PITCH_LENGTH, grid_res[0])
    y = np.linspace(0, PITCH_WIDTH, grid_res[1])
    grid_x, grid_y = np.meshgrid(x, y)
    
    def min_arrival_time(positions):
        if len(positions) == 0: return np.full(grid_x.shape, 999.0)
        times = [reaction_time + (np.sqrt((grid_x - px)**2 + (grid_y - py)**2) / v_max) for px, py in positions]
        return np.minimum.reduce(times)

    t_home = min_arrival_time(home_pos)
    t_away = min_arrival_time(away_pos)
    return grid_x, grid_y, 1.0 / (1.0 + np.exp(4.0 * (t_home - t_away)))

def calculate_passing_lanes(passer_pos, teammates, opponents):
    lanes = []
    px, py = passer_pos
    for tx, ty in teammates:
        if (tx, ty) == (px, py): continue
        dist = np.sqrt((tx - px)**2 + (ty - py)**2)
        if dist > 35.0 or dist < 3.0: continue
        
        intercept_risk = False
        for ox, oy in opponents:
            line_vec = np.array([tx - px, ty - py])
            line_len = np.linalg.norm(line_vec)
            if line_len == 0: continue
            line_unit = line_vec / line_len
            def_vec = np.array([ox - px, oy - py])
            proj_len = np.dot(def_vec, line_unit)
            
            if 0 < proj_len < line_len:
                perp_dist = np.linalg.norm(def_vec - proj_len * line_unit)
                if perp_dist < 2.5:
                    intercept_risk = True; break
        lanes.append(((px, py), (tx, ty), intercept_risk))
    return lanes

# ---------------------------------------------------------
# 3. STREAMLIT UI & PLAYBACK CONTROLS
# ---------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg", width=70)
st.title("⚽ Arsenal Pro AI — Real-Time Playback Engine")

# Playback Sidebar Controls
st.sidebar.header("⏱️ Real-Time Match Player")
btn_col1, btn_col2 = st.sidebar.columns(2)

if btn_col1.button("▶️ Play Match" if not st.session_state.is_playing else "⏸️ Pause"):
    st.session_state.is_playing = not st.session_state.is_playing

if btn_col2.button("⏮️ Reset"):
    st.session_state.current_frame = 100
    st.session_state.is_playing = False

playback_speed = st.sidebar.select_slider(
    "Playback Speed", options=[0.5, 1.0, 1.5, 2.0], value=1.0, format_func=lambda x: f"{x}x Speed"
)

frame_idx = st.sidebar.slider(
    "Scrub Timeline Frame", 1, 1500, int(st.session_state.current_frame)
)
st.session_state.current_frame = frame_idx

show_physics = st.sidebar.checkbox("Kinematic Physics Heatmap", True)
show_pass_lanes = st.sidebar.checkbox("Passing Lane Interception Radar", True)

# Extract Positions for Current Frame
home_pos, away_pos = [], []
for col in df_home.columns:
    if col.endswith('_x') and 'Ball' not in col:
        p_id = col.replace('Home_', '').replace('_x', '')
        hx = df_home[f'Home_{p_id}_x'].iloc[frame_idx] * PITCH_LENGTH
        hy = df_home[f'Home_{p_id}_y'].iloc[frame_idx] * PITCH_WIDTH
        if not pd.isna(hx) and not pd.isna(hy): home_pos.append((hx, hy))

for col in df_away.columns:
    if col.endswith('_x') and 'Ball' not in col:
        p_id = col.replace('Away_', '').replace('_x', '')
        ax_p = df_away[f'Away_{p_id}_x'].iloc[frame_idx] * PITCH_LENGTH
        ay_p = df_away[f'Away_{p_id}_y'].iloc[frame_idx] * PITCH_WIDTH
        if not pd.isna(ax_p) and not pd.isna(ay_p): away_pos.append((ax_p, ay_p))

ball_x = df_home['Ball_x'].iloc[frame_idx] * PITCH_LENGTH
ball_y = df_home['Ball_y'].iloc[frame_idx] * PITCH_WIDTH

def_line_height = np.min([x for x, y in home_pos]) if home_pos else 0

# Metric Banner
m1, m2, m3, m4 = st.columns(4)
m1.metric("Match Time", f"{df_home['Time'].iloc[frame_idx]:.2f}s")
m2.metric("FPS Mode", "25 Hz Tracking")
m3.metric("Defensive Line Height", f"{def_line_height:.1f}m")
m4.metric("Engine Status", "LIVE MATCH STREAMING" if st.session_state.is_playing else "PAUSED")

# ---------------------------------------------------------
# 4. DRAW ANIMATED PITCH SNAPSHOT
# ---------------------------------------------------------
pitch = Pitch(pitch_type='custom', pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
              pitch_color='#11161d', line_color='#ffffff')
fig, ax = pitch.draw(figsize=(12, 6.8))

if show_physics:
    gx, gy, control_map = compute_kinematic_pitch_control(home_pos, away_pos)
    ax.contourf(gx, gy, control_map, levels=10, cmap='coolwarm', alpha=0.35, zorder=1)

if show_pass_lanes and not pd.isna(ball_x):
    passer = min(home_pos, key=lambda p: (p[0]-ball_x)**2 + (p[1]-ball_y)**2) if home_pos else None
    if passer:
        lanes = calculate_passing_lanes(passer, home_pos, away_pos)
        for (start, end, blocked) in lanes:
            color = '#ff4d4d' if blocked else '#2ec4b6'
            ls = '--' if blocked else '-'
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linestyle=ls, linewidth=2.5, zorder=2)

hx_v, hy_v = zip(*home_pos) if home_pos else ([], [])
ax_v, ay_v = zip(*away_pos) if away_pos else ([], [])

pitch.scatter(hx_v, hy_v, s=260, color='#e63946', edgecolors='white', linewidth=2, ax=ax, zorder=4)
pitch.scatter(ax_v, ay_v, s=260, color='#1d3557', edgecolors='white', linewidth=2, ax=ax, zorder=4)

if not pd.isna(ball_x) and not pd.isna(ball_y):
    pitch.scatter(ball_x, ball_y, s=150, color='#ffee32', edgecolors='black', linewidth=2, ax=ax, zorder=5)

st.pyplot(fig)

# ---------------------------------------------------------
# 5. REAL-TIME LOOP DRIVER
# ---------------------------------------------------------
if st.session_state.is_playing:
    # Advance frame index based on playback speed (25fps base rate)
    st.session_state.current_frame += int(1 * playback_speed)
    if st.session_state.current_frame >= 1500:
        st.session_state.current_frame = 100
        st.session_state.is_playing = False
    
    # Refresh app state smoothly
    time.sleep(0.04 / playback_speed)
    st.rerun()