import psycopg2

def main():
    conn = psycopg2.connect("dbname=vecdb")
    cur = conn.cursor()

    # Clean table
    cur.execute("DELETE FROM vectors;")
    conn.commit()

    # Insert two small vectors
    cur.execute(
        "INSERT INTO vectors (embedding, cls) VALUES (%s, %s)",
        ([0.1, 0.2, 0.3], "A"),
    )
    cur.execute(
        "INSERT INTO vectors (embedding, cls) VALUES (%s, %s)",
        ([0.0, 0.0, 1.0], "B"),
    )
    conn.commit()

    # Query by similarity (L2 distance)
    cur.execute(
        """
        SELECT id, cls, embedding <-> %s::vector AS dist
        FROM vectors
        ORDER BY embedding <-> %s::vector
        LIMIT 2;
        """,
        ([0.1, 0.2, 0.25], [0.1, 0.2, 0.25])
    )


    rows = cur.fetchall()
    print("pgvector smoke test result:")
    for row in rows:
        print(row)

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
