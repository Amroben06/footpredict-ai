import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# 1. LOAD & PREPROCESS TRACKING DATA
# ---------------------------------------------------------
print("⏳ Downloading Metrica tracking data...")
url_home = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
df_home = pd.read_csv(url_home, skiprows=2)

# Helper function to clean columns
def clean_metrica(df):
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
            new_cols.append(f'P{p_id}_{axis}')
    df.columns = new_cols
    return df

df_home = clean_metrica(df_home)

# Extract spatial coordinate features (dropping metadata columns)
feature_cols = [c for c in df_home.columns if c not in ['Period', 'Frame', 'Time']]
data_matrix = df_home[feature_cols].fillna(0).values  # Matrix of shape (Frames, Num_Coordinates)

# Convert to PyTorch Tensor
X = torch.tensor(data_matrix, dtype=torch.float32)

# Dummy target: In a full pipeline, this comes from event outcomes (e.g., shot in next 10s).
# Here, we derive threat synthetically based on ball proximity to the opponent's goal area (x > 0.8)
ball_x_index = feature_cols.index('Ball_x')
y_threat = (X[:, ball_x_index] > 0.8).float().unsqueeze(1)

print(f"✅ Data prepared: {X.shape[0]} match frames with {X.shape[1]} spatial features per frame.")


# ---------------------------------------------------------
# 2. DEFINE THE NEURAL NETWORK ARCHITECTURE
# ---------------------------------------------------------
class FootballTacticsEvaluator(nn.Module):
    def __init__(self, input_dim):
        super(FootballTacticsEvaluator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability between 0.0 and 1.0
        )

    def forward(self, x):
        return self.net(x)

model = FootballTacticsEvaluator(input_dim=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------
# 3. TRAIN THE AI MODEL
# ---------------------------------------------------------
print("\n🏋️ Training Football State Evaluator AI...")
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y_threat)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] — Loss: {loss.item():.4f}")

print("✅ Training complete!")

# ---------------------------------------------------------
# 4. INFERENCE: EVALUATE A SPECIFIC MATCH FRAME
# ---------------------------------------------------------
test_frame = 100
sample_input = X[test_frame].unsqueeze(0)
model.eval()
with torch.no_grad():
    threat_score = model(sample_input).item()

print(f"\n🎯 Frame {test_frame} AI Tactical Threat Valuation: {threat_score * 100:.2f}%")