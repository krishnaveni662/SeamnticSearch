# from pymilvus import MilvusClient, DataType

# # Connect to Milvus
# client = MilvusClient("http://localhost:19530")
# print("✅ Connected to Milvus!")

# # List collections (should be empty initially)
# print("Collections:", client.list_collections())

# # Create sample collection
# client.create_collection(
#     collection_name="test_vectors",
#     dimension=128,  # Vector size
#     metric_type="L2"
# )
# print("✅ Collection 'test_vectors' created!")

# 1. Installation:
# pip install sentence-transformers torch

from sentence_transformers import SentenceTransformer

# 2. Choose your model (e.g., E5-base-v2)
MODEL_NAME = 'intfloat/e5-base-v2' 

# 3. Load the model locally (only happens once)
model = SentenceTransformer(MODEL_NAME, device='cpu') 

# 4. Embed your texts
texts_to_embed = ["What is the capital of France?", "Paris is the capital city."]
embeddings = model.encode(texts_to_embed, convert_to_tensor=False)

# embeddings is a numpy array containing your vectors
