import chromadb

def pad3(dim, a, b, c):
    v = [0.0] * dim
    v[0], v[1], v[2] = a, b, c
    return v

def main():
    DIM = 100
    client = chromadb.HttpClient(host="localhost", port=8000)

    #Delete old collection if exists
    try:
        client.delete_collection("smoke")
    except Exception:
        pass

    collection = client.get_or_create_collection("smoke")

    v1 = pad3(DIM, 0.1, 0.2, 0.3)
    v2 = pad3(DIM, 0.0, 0.0, 0.9)
    q  = pad3(DIM, 0.1, 0.2, 0.25)

    collection.add(
        ids=["1", "2"],
        embeddings=[v1, v2],
        metadatas=[{"cls": "A"}, {"cls": "B"}],
    )

    result = collection.query(
        query_embeddings=[q],
        n_results=2,
        include=["metadatas", "distances"],
    )

    print("Chroma smoke test result:")
    ids = result["ids"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for i in range(len(ids)):
        print(ids[i], metas[i]["cls"], dists[i])

if __name__ == "__main__":
    main()
