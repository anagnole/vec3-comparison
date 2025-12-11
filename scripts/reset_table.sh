#!/bin/bash

CONTAINER=pgvector_bench   # <-- update this to match your container name
DB=vecdb


echo "Resetting pgvector table inside Docker Postgres..."

docker exec -i $CONTAINER psql -U user -d $DB <<EOF
DROP TABLE IF EXISTS vectors;

CREATE TABLE vectors (
    id BIGSERIAL PRIMARY KEY,
    embedding vector(100),  -- update dim if needed
    cls TEXT
);
EOF

echo "Done."