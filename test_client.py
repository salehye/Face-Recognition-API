import requests
import json
import base64
import os

API_URL = "http://localhost:8001"

def print_section(title):
    print("=" * 60)
    print(f" {title} ")
    print("=" * 60)

def test_root():
    print_section("1. Testing Root Endpoint")
    res = requests.get(f"{API_URL}/")
    print(f"Status: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2, ensure_ascii=False)}")
    assert res.status_code == 200

def test_detect():
    print_section("2. Testing Face Detection (/detect)")
    with open("test_images/alice_1.png", "rb") as f:
        files = {"file": f}
        res = requests.post(f"{API_URL}/detect", files=files)
    print(f"Status: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}")
    assert res.status_code == 200
    assert res.json()["count"] > 0

def test_detect_embeddings():
    print_section("2b. Testing Face Detection with Embeddings (/detect-embeddings)")
    with open("test_images/alice_1.png", "rb") as f:
        files = {"file": f}
        res = requests.post(f"{API_URL}/detect-embeddings", files=files)
    print(f"Status: {res.status_code}")
    data = res.json()
    if "faces" in data and len(data["faces"]) > 0:
        face = data["faces"][0]
        emb_preview = face["embedding"][:5]
        print(f"Detected {data['count']} face(s). First face embedding preview (first 5 values): {emb_preview}...")
        print(f"Bounding Box: {face['bbox']}")
        print(f"Confidence: {face['confidence']}")
    else:
        print("No faces detected in response.")
    assert res.status_code == 200
    assert data["count"] > 0

def test_embeddings():
    print_section("3. Testing Embedding Extraction (/embedding)")
    
    # Alice 1
    with open("test_images/alice_1.png", "rb") as f:
        res1 = requests.post(f"{API_URL}/embedding", files={"file": f})
    alice_1_data = res1.json()
    print("Alice 1 Embedding Size:", alice_1_data["embedding_size"])
    print("Alice 1 Confidence:", alice_1_data["confidence"])
    
    # Bob
    with open("test_images/bob.png", "rb") as f:
        res2 = requests.post(f"{API_URL}/embedding", files={"file": f})
    bob_data = res2.json()
    print("Bob Confidence:", bob_data["confidence"])

    # Alice 2
    with open("test_images/alice_2.png", "rb") as f:
        res3 = requests.post(f"{API_URL}/embedding", files={"file": f})
    alice_2_data = res3.json()
    print("Alice 2 Confidence:", alice_2_data["confidence"])

    return alice_1_data["embedding"], bob_data["embedding"], alice_2_data["embedding"]

def test_base64():
    print_section("4. Testing Base64 Embedding (/embedding-base64)")
    with open("test_images/alice_1.png", "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {"image_base64": f"data:image/png;base64,{encoded}"}
    res = requests.post(f"{API_URL}/embedding-base64", json=payload)
    print(f"Status: {res.status_code}")
    data = res.json()
    print("Success:", data["success"])
    print("Confidence:", data["confidence"])
    assert res.status_code == 200
    assert data["success"] is True

def test_compare_embeddings(emb_alice1, emb_bob, emb_alice2):
    print_section("5. Testing Embedding Comparison (/compare)")
    
    # Compare Alice 1 & Alice 2 (Should match)
    payload_match = {
        "emb1": emb_alice1,
        "emb2": emb_alice2,
        "threshold": 0.50
    }
    res_match = requests.post(f"{API_URL}/compare", json=payload_match)
    match_data = res_match.json()
    print(f"Alice 1 vs Alice 2 (Expected Match):")
    print(f"  Similarity: {match_data['similarity']:.4f}")
    print(f"  Is Match: {match_data['is_match']}")
    
    # Compare Alice 1 & Bob (Should not match)
    payload_diff = {
        "emb1": emb_alice1,
        "emb2": emb_bob,
        "threshold": 0.50
    }
    res_diff = requests.post(f"{API_URL}/compare", json=payload_diff)
    diff_data = res_diff.json()
    print(f"Alice 1 vs Bob (Expected Non-Match):")
    print(f"  Similarity: {diff_data['similarity']:.4f}")
    print(f"  Is Match: {diff_data['is_match']}")

def test_compare_files():
    print_section("6. Testing File Comparison (/compare-files)")
    with open("test_images/alice_1.png", "rb") as f1, open("test_images/alice_2.png", "rb") as f2:
        files = {
            "file1": f1,
            "file2": f2
        }
        res = requests.post(f"{API_URL}/compare-files?threshold=0.50", files=files)
    print(f"Status: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}")
    assert res.status_code == 200
    assert res.json()["is_match"] is True

def test_search(emb_alice1, emb_bob, emb_alice2):
    print_section("7. Testing Database Search (/search)")
    
    # Database of embeddings (indices: 0 is Bob, 1 is Alice 2)
    db = [emb_bob, emb_alice2]
    
    payload = {
        "query_embedding": emb_alice1,
        "database_embeddings": db,
        "threshold": 0.50,
        "top_k": 5
    }
    
    res = requests.post(f"{API_URL}/search", json=payload)
    print(f"Status: {res.status_code}")
    data = res.json()
    print(f"Search Results: {json.dumps(data, indent=2)}")
    
    assert res.status_code == 200
    # Should match Alice 2 (index 1) with high similarity
    matches = data["matches"]
    assert len(matches) > 0
    assert matches[0]["index"] == 1  # Alice 2 should be the top match

if __name__ == "__main__":
    if not os.path.exists("test_images"):
        print("Error: test_images folder not found. Please place images in it.")
        exit(1)
        
    try:
        test_root()
        test_detect()
        test_detect_embeddings()
        emb_alice1, emb_bob, emb_alice2 = test_embeddings()
        test_base64()
        test_compare_embeddings(emb_alice1, emb_bob, emb_alice2)
        test_compare_files()
        test_search(emb_alice1, emb_bob, emb_alice2)
        print("\nAll tests completed successfully!")
    except Exception as e:
        print("\nAn error occurred during testing:")
        import traceback
        traceback.print_exc()
