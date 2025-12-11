"""
Script to manually create the Milvus collection.
Run this if the collection doesn't exist or if you need to recreate it.
"""
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import time

load_dotenv()
HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION = "docs"
DIM = 768  # bge-base outputs 768 dims; CLIP is padded to 768

def create_collection():
    """Create the collection if it doesn't exist."""
    # Connect to Milvus
    try:
        connections.connect(alias="default", host=HOST, port=PORT)
        print(f"Connected to Milvus at {HOST}:{PORT}")
    except Exception as e:
        print(f"ERROR: Could not connect to Milvus at {HOST}:{PORT}")
        print(f"Error details: {e}")
        print("\nMake sure Milvus is running!")
        print("Start Milvus with: docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.10-standalone")
        return
    
    # Try to drop collection if it exists (ignore errors)
    print("Attempting to drop existing collection if it exists...")
    try:
        utility.drop_collection(COLLECTION)
        print(f"Dropped existing collection '{COLLECTION}'")
        time.sleep(2)  # Wait for drop to complete
    except Exception:
        # Collection doesn't exist, which is fine
        print("No existing collection found (or already dropped).")
        pass
    
    # Define schema fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="meta_classification", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="meta_source", dtype=DataType.VARCHAR, max_length=256),
    ]
    schema = CollectionSchema(fields, description="docs with RBAC")
    
    # Create collection directly - don't check existence first
    print(f"Creating collection '{COLLECTION}' with dimension {DIM}...")
    try:
        # Create collection directly - this should work even if has_collection fails
        coll = Collection(COLLECTION, schema, consistency_level="Strong")
        print(f"✓ Collection '{COLLECTION}' created successfully!")
    except Exception as e:
        error_msg = str(e)
        # Check if collection was actually created despite the error
        try:
            coll = Collection(COLLECTION)
            print(f"✓ Collection '{COLLECTION}' exists (created despite error message)")
        except:
            print(f"✗ ERROR: Failed to create collection: {error_msg}")
            print("\nTroubleshooting:")
            print("1. Try: docker restart milvus-standalone")
            print("2. Check: docker logs milvus-standalone | Select-String -Pattern error")
            print("3. Consider upgrading pymilvus: pip install --upgrade pymilvus")
            return
    
    # Create index
    print("Creating index...")
    coll.create_index("embedding", {"index_type":"IVF_FLAT","metric_type":"IP","params":{"nlist":1024}})
    print("Index created successfully!")
    
    # Load collection
    coll.load()
    print(f"Collection '{COLLECTION}' is loaded and ready to use!")
    
    # Print collection info
    print(f"\nCollection info:")
    print(f"  Name: {coll.name}")
    print(f"  Schema fields: {[f.name for f in coll.schema.fields]}")
    print(f"  Number of entities: {coll.num_entities}")

if __name__ == "__main__":
    try:
        create_collection()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Milvus is running!")
        print("Start Milvus with: docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.10-standalone")

