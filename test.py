from pymilvus import MilvusClient, DataType

# Connect to Milvus
client = MilvusClient("http://localhost:19530")
print("✅ Connected to Milvus!")

# List collections (should be empty initially)
print("Collections:", client.list_collections())

# Create sample collection
client.create_collection(
    collection_name="test_vectors",
    dimension=128,  # Vector size
    metric_type="L2"
)
print("✅ Collection 'test_vectors' created!")
