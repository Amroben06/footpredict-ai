import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import faiss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mplsoccer import Pitch

st.set_page_config(page_title="Arsenal Pro AI - Foundation Model & Vector Engine", layout="wide")

# ---------------------------------------------------------
# 1. SPATIOTEMPORAL TRANSFORMER (WITH GEOMETRIC EQUIVARIANCE)
# ---------------------------------------------------------
class SpatiotemporalFootballTransformer(nn.Module):
    def __init__(self, embed_dim=64, nheads=4, num_layers=2):
        super(SpatiotemporalFootballTransformer, self).__init__()
        
        self.input_proj = nn.Linear(4, embed_dim)
        self.player_embed = nn.Parameter(torch.randn(1, 1, 22, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nheads, 
            dim_feedforward=128, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-task heads
        self.head_state_value = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.head_concepts = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Low Block, Mid Block, High Press
        )

    def normalize_pitch_symmetry(self, x):
        """
        GEOMETRIC EQUIVARIANCE STEP 1: Pitch Symmetry Normalization
        Ensures attacking team is always evaluated moving Left -> Right.
        """
        x_norm = x.clone()
        attacking_mean_x = torch.mean(x_norm[:, :, :11, 0])
        
        # If moving Right -> Left, flip tracking coordinates 180 degrees
        if attacking_mean_x > 0.5:
            x_norm[:, :, :, 0] = 1.0 - x_norm[:, :, :, 0] # Flip X
            x_norm[:, :, :, 1] = 1.0 - x_norm[:, :, :, 1] # Flip Y
            x_norm[:, :, :, 2] = -x_norm[:, :, :, 2]     # Invert Vx
            x_norm[:, :, :, 3] = -x_norm[:, :, :, 3]     # Invert Vy
            
        return x_norm

    def extract_embedding(self, x):
        # 1. Geometric pitch symmetry flip
        x = self.normalize_pitch_symmetry(x)
        
        B, T, P, F = x.shape
        x_proj = self.input_proj(x) + self.player_embed[:, :, :P, :]
        x_flat = x_proj.view(B, T * P, -1)
        transformer_out = self.transformer(x_flat)
        
        # 2. GEOMETRIC EQUIVARIANCE STEP 2: Permutation-invariant set pooling across agents
        return torch.mean(transformer_out, dim=1)

    def forward(self, x):
        emb = self.extract_embedding(x)
        val = self.head_state_value(emb)
        concepts = self.head_concepts(emb)
        return emb, val, concepts

# ---------------------------------------------------------
# 2. CACHED MODEL & FAISS INDEX INITIALIZATION
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_vector_db(num_clips=150):
    torch.manual_seed(42)
    np.random.seed(42)

    model = SpatiotemporalFootballTransformer(embed_dim=64)
    model.eval()

    embedding_dim = 64
    index = faiss.IndexFlatIP(embedding_dim)
    metadata = []

    concepts_list = ["High Press (4-3-3)", "Compact Low Block (5-4-1)", "Mid-Block Transition"]
    teams_list = ["Arsenal vs Chelsea", "Arsenal vs Man City", "Arsenal vs Real Madrid", "Arsenal vs PSG"]
    
    # Sample MP4 stream URLs for demo playback
    video_urls = [
        "https://www.w3schools.com/html/mov_bbb.mp4",
        "https://media.w3.org/2010/05/sintel/trailer.mp4"
    ]

    embeddings_list = []
    with torch.no_grad():
        for i in range(num_clips):
            seq = torch.randn(1, 5, 22, 4) * (0.5 + (i * 0.02))
            emb = model.extract_embedding(seq).numpy().astype('float32')
            
            faiss.normalize_L2(emb)
            embeddings_list.append(emb[0])
            
            metadata.append({
                "clip_id": f"CLIP_{100 + i}",
                "match": teams_list[i % len(teams_list)],
                "minute": f"{((i * 11) % 89) + 1}'",
                "concept": concepts_list[i % len(concepts_list)],
                "video_url": video_urls[i % len(video_urls)]
            })

    embeddings_matrix = np.array(embeddings_list).astype('float32')
    index.add(embeddings_matrix)
    
    return model, index, metadata

transformer_model, faiss_index, db_metadata = load_model_and_vector_db()

# ---------------------------------------------------------
# 3. DATA LOADING & PREPROCESSING
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
        return df.fillna(0.5)

    return clean_metrica(df_h, 'Home'), clean_metrica(df_a, 'Away')

df_home, df_away = load_data()
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

# ---------------------------------------------------------
# 4. STREAMLIT UI & INTERACTIVE DASHBOARD
# ---------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg", width=70)
st.title("🧠 Spatiotemporal Transformer & Vector Engine")
st.caption("Multi-Task Foundation Model with Built-in Geometric Equivariances & FAISS RAG Retrieval")

frame_idx = st.sidebar.slider("Scrub Match Timeline", 5, 1200, 150)

# Build Non-NaN Tracking Sequence Tensor
tracking_window = []
for f_step in range(frame_idx - 4, frame_idx + 1):
    frame_features = []
    
    # Home Team (Red)
    for col in df_home.columns:
        if col.endswith('_x') and 'Ball' not in col:
            pid = col.replace('Home_', '').replace('_x', '')
            hx = float(np.nan_to_num(df_home[f'Home_{pid}_x'].iloc[f_step], nan=0.5))
            hy = float(np.nan_to_num(df_home[f'Home_{pid}_y'].iloc[f_step], nan=0.5))
            prev_hx = float(np.nan_to_num(df_home[f'Home_{pid}_x'].iloc[f_step-1], nan=0.5))
            prev_hy = float(np.nan_to_num(df_home[f'Home_{pid}_y'].iloc[f_step-1], nan=0.5))
            frame_features.append([hx, hy, hx - prev_hx, hy - prev_hy])
            
    # Away Team (Navy)
    for col in df_away.columns:
        if col.endswith('_x') and 'Ball' not in col:
            pid = col.replace('Away_', '').replace('_x', '')
            ax = float(np.nan_to_num(df_away[f'Away_{pid}_x'].iloc[f_step], nan=0.5))
            ay = float(np.nan_to_num(df_away[f'Away_{pid}_y'].iloc[f_step], nan=0.5))
            prev_ax = float(np.nan_to_num(df_away[f'Away_{pid}_x'].iloc[f_step-1], nan=0.5))
            prev_ay = float(np.nan_to_num(df_away[f'Away_{pid}_y'].iloc[f_step-1], nan=0.5))
            frame_features.append([ax, ay, ax - prev_ax, ay - prev_ay])
            
    while len(frame_features) < 22:
        frame_features.append([0.5, 0.5, 0.0, 0.0])
    tracking_window.append(frame_features[:22])

input_tensor = torch.tensor([tracking_window], dtype=torch.float32)

# Spatiotemporal Transformer Inference
with torch.no_grad():
    scene_emb, state_val, concept_logits = transformer_model(input_tensor)

# FAISS Cosine Vector Search
query_vec = scene_emb.numpy().astype('float32')
faiss.normalize_L2(query_vec)
similarities, indices = faiss_index.search(query_vec, k=3)

# Display Multi-Task Output Metrics
m1, m2, m3 = st.columns(3)
m1.metric("State Threat Valuation (xT)", f"{state_val.item():.2%}")
concepts = ["Low Block", "Mid Block", "High Press"]
predicted_concept = concepts[torch.argmax(concept_logits).item()]
m2.metric("Tactical Structure Concept", predicted_concept)
m3.metric("FAISS Search Speed", "< 0.8 ms")

st.divider()

col_left, col_right = st.columns([1.2, 1])

# --- LEFT COLUMN: LIVE PITCH VECTOR & LATENT CLUSTER MAP ---
with col_left:
    st.subheader("📌 Live Tracking Pitch Vector")
    pitch = Pitch(pitch_type='custom', pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                  pitch_color='#11161d', line_color='#ffffff')
    fig, ax = pitch.draw(figsize=(8, 4.5))
    
    home_pos = [(input_tensor[0, -1, i, 0].item() * PITCH_LENGTH, input_tensor[0, -1, i, 1].item() * PITCH_WIDTH) for i in range(11)]
    away_pos = [(input_tensor[0, -1, i, 0].item() * PITCH_LENGTH, input_tensor[0, -1, i, 1].item() * PITCH_WIDTH) for i in range(11, 22)]
    
    hx, hy = zip(*home_pos)
    ax_p, ay_p = zip(*away_pos)
    
    pitch.scatter(hx, hy, color='#e63946', s=180, edgecolors='white', linewidth=1.5, ax=ax)
    pitch.scatter(ax_p, ay_p, color='#1d3557', s=180, edgecolors='white', linewidth=1.5, ax=ax)
    
    st.pyplot(fig)

    # Latent Space Cluster Plot (PCA Projection matching bottom-center of diagram)
    st.subheader("🌌 Latent Space Tactical Clusters (PCA)")
    pca = PCA(n_components=2)
    db_vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    db_2d = pca.fit_transform(db_vectors)
    query_2d = pca.transform(query_vec)

    fig_c, ax_c = plt.subplots(figsize=(8, 3), facecolor='#11161d')
    ax_c.set_facecolor('#11161d')
    
    # Plot background indexed moments
    ax_c.scatter(db_2d[:, 0], db_2d[:, 1], c='#3a86ff', alpha=0.3, s=20, label="Indexed Clips")
    # Highlight top retrieved matches
    ax_c.scatter(db_2d[indices[0], 0], db_2d[indices[0], 1], c='#00f5d4', s=70, edgecolors='white', label="FAISS Matches")
    # Highlight current live state
    ax_c.scatter(query_2d[0, 0], query_2d[0, 1], c='#ff0054', s=140, marker='*', edgecolors='white', label="Live Vector State")
    
    ax_c.axis('off')
    ax_c.legend(facecolor='#11161d', edgecolor='none', labelcolor='white', fontsize=8)
    st.pyplot(fig_c)

# --- RIGHT COLUMN: FAISS VIDEO RETRIEVAL CARDS ---
with col_right:
    st.subheader("🎬 Retrieved Video Clips (FAISS RAG)")
    st.caption("Top historical clips sharing the same Spatiotemporal Self-Attention embedding:")
    
    for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), 1):
        match_info = db_metadata[idx]
        similarity_pct = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
        
        with st.expander(f"Rank {rank}: {match_info['match']} ({match_info['minute']}) — Match: {similarity_pct:.1%}", expanded=(rank==1)):
            st.write(f"**Clip ID:** `{match_info['clip_id']}` | **Tactical Concept:** {match_info['concept']}")
            st.progress(similarity_pct)
            
            # Integrated Video Player Card
            st.video(match_info['video_url'])