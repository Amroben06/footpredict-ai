import torch
import torch.nn as nn
import numpy as np

class SpatiotemporalFootballTransformer(nn.Module):
    """
    Multi-task Spatiotemporal Transformer Foundation Model for Football Analytics.
    Inputs: Tracking sequence tensor of shape (Batch, Frames, 22_Players, Features)
    """
    def __init__(self, embed_dim=128, nheads=4, num_layers=3, num_concepts=5):
        super(SpatiotemporalFootballTransformer, self).__init__()
        
        # 1. Feature Projection (Input: X, Y, Vx, Vy per player = 4 features)
        self.input_projection = nn.Linear(4, embed_dim)
        
        # 2. Player & Temporal Positional Embeddings
        self.player_embed = nn.Parameter(torch.randn(1, 1, 22, embed_dim))
        
        # 3. Spatiotemporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nheads, 
            dim_feedforward=256, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Multi-Task Output Heads
        # Head A: State Valuation (Expected Threat / Danger Score)
        self.head_state_value = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head B: Structured Football Concepts (e.g., High Press, Low Block, Counter-attack)
        self.head_concepts = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_concepts)
        )
        
        # Head C: Action Valuation (Score for every player's next move)
        self.head_action_value = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def extract_embedding(self, x):
        """
        x shape: (Batch, Frames, 22, 4)
        Returns a single dense vector embedding (Batch, Embed_Dim) representing the tactical state.
        """
        B, T, P, F = x.shape
        
        # Project raw features to embedding space
        x_proj = self.input_projection(x) + self.player_embed  # (B, T, P, Dim)
        
        # Flatten spatial & temporal dimensions for Transformer Self-Attention
        x_flat = x_proj.view(B, T * P, -1)  # (B, T*22, Dim)
        
        # Pass through Transformer
        transformer_out = self.transformer(x_flat)  # (B, T*22, Dim)
        
        # Pooling across all agents and time steps to get scene embedding
        scene_embedding = torch.mean(transformer_out, dim=1)  # (B, Dim)
        return scene_embedding

    def forward(self, x):
        # Extract bottleneck representation
        scene_embedding = self.extract_embedding(x)
        
        # Multi-task outputs
        state_val = self.head_state_value(scene_embedding)
        concept_logits = self.head_concepts(scene_embedding)
        
        return {
            "scene_embedding": scene_embedding,
            "state_valuation": state_val,
            "concept_logits": concept_logits
        }

# --- Quick Test Output ---
if __name__ == "__main__":
    model = SpatiotemporalFootballTransformer(embed_dim=128)
    
    # Simulate a batch of 4 play clips, each 10 frames long, 22 players, 4 features (x, y, vx, vy)
    dummy_input = torch.randn(4, 10, 22, 4)
    outputs = model(dummy_input)
    
    print("✅ Spatiotemporal Transformer initialized successfully!")
    print(f"  • Scene Embedding Shape: {outputs['scene_embedding'].shape}")
    print(f"  • State Valuation Shape: {outputs['state_valuation'].shape}")
    print(f"  • Concept Logits Shape:  {outputs['concept_logits'].shape}")