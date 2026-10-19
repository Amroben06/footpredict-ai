import numpy as np
import torch
import faiss
from transformer_engine import SpatiotemporalFootballTransformer

class TacticalRetrievalEngine:
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        # Initialize L2 Euclidean Distance FAISS Index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []  # Stores corresponding match timestamps and clip IDs

    def build_index(self, model, synthetic_num_clips=500):
        """Generates embeddings for historical tracking clips and indexes them in FAISS."""
        print(f"⚡ Indexing {synthetic_num_clips} historical tactical moments...")
        
        model.eval()
        embeddings_list = []
        
        with torch.no_grad():
            for i in range(synthetic_num_clips):
                # Simulate historical 10-frame clip
                clip_tensor = torch.randn(1, 10, 22, 4)
                emb = model.extract_embedding(clip_tensor).numpy()
                
                embeddings_list.append(emb[0])
                
                # Store clip metadata
                self.metadata.append({
                    "clip_id": f"CLIP_{1000 + i}",
                    "match": "Arsenal vs Man Utd",
                    "timestamp": f"{np.random.randint(1, 90)}:{np.random.randint(10, 59)}"
                })

        # Add all vectors to FAISS index
        embeddings_matrix = np.array(embeddings_list).astype('float32')
        self.index.add(embeddings_matrix)
        print(f"✅ Indexed {self.index.ntotal} vectors successfully in FAISS!")

    def query_similar_moments(self, query_embedding, top_k=3):
        """Finds the top-K most similar historical clips for a given tactical state."""
        query_vector = query_embedding.detach().numpy().astype('float32')
        
        # Search vector index
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            match_info = self.metadata[idx]
            match_info["similarity_score"] = float(1.0 / (1.0 + dist)) # Convert distance to similarity
            results.append(match_info)
            
        return results

# --- Test Vector Search ---
if __name__ == "__main__":
    # 1. Instantiate Transformer Foundation Model
    transformer_model = SpatiotemporalFootballTransformer(embed_dim=128)
    
    # 2. Build FAISS Index
    search_engine = TacticalRetrievalEngine(embedding_dim=128)
    search_engine.build_index(transformer_model, synthetic_num_clips=200)
    
    # 3. Query with a Live Frame
    live_clip = torch.randn(1, 10, 22, 4)
    live_embedding = transformer_model.extract_embedding(live_clip)
    
    top_matches = search_engine.query_similar_moments(live_embedding, top_k=3)
    
    print("\n🔍 Top 3 Historical Clips Matching Current Tactical Setup:")
    for rank, match in enumerate(top_matches, 1):
        print(f"  Rank {rank}: [{match['clip_id']}] {match['match']} at {match['timestamp']} (Match Score: {match['similarity_score']:.2%})")