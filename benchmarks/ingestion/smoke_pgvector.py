import os
import psycopg2

def get_conn():
    #Default: Docker-compose Postgres
    dsn = os.getenv("PG_DSN", "host=localhost port=5432 dbname=vecdb user=user")
    return psycopg2.connect(dsn)

def main():
    conn = get_conn()
    cur = conn.cursor()

    #embedding vector(100)
    DIM = 100

    def pad3(a, b, c):
        v = [0.0] * DIM
        v[0], v[1], v[2] = a, b, c
        return v

    v1 = pad3(0.1, 0.2, 0.3)
    v2 = pad3(0.0, 0.0, 1.0)
    q  = pad3(0.1, 0.2, 0.25)

    # Clean table
    cur.execute("DELETE FROM vectors;")
    conn.commit()

    # Insert two small vectors
    cur.execute(
        "INSERT INTO vectors (embedding, cls) VALUES (%s, %s)",
        (v1, "A"),
    )
    cur.execute(
        "INSERT INTO vectors (embedding, cls) VALUES (%s, %s)",
        (v2, "B"),
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
        (q, q)
    )


    rows = cur.fetchall()
    print("pgvector smoke test result:")
    for row in rows:
        print(row)

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
