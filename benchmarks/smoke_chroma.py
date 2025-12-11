import chromadb

def main():
    client = chromadb.HttpClient(host="localhost", port=8000)

    collection = client.get_or_create_collection("smoke")

    collection.add(
        ids=["1", "2"],
        embeddings=[[0.1, 0.2, 0.3], [0.0, 0.0, 0.9]],
        metadatas=[{"cls": "A"}, {"cls": "B"}],
    )

    result = collection.query(
        query_embeddings=[[0.1, 0.2, 0.25]],
        n_results=2,
    )

    print("Chroma smoke test result:")
    print(result)

if __name__ == "__main__":
    main()
