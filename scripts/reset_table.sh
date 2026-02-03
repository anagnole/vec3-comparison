#!/bin/bash

CONTAINER=pgvector_bench   # <-- update this to match your container name
DB=vecdb


echo "Resetting pgvector tables inside Docker Postgres..."

docker exec -i $CONTAINER psql -U user -d $DB <<EOF
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS vectors_10k;
DROP TABLE IF EXISTS vectors_100k;
DROP TABLE IF EXISTS vectors_200k;
DROP TABLE IF EXISTS vectors_500k;

CREATE TABLE vectors_10k (
    id BIGSERIAL PRIMARY KEY,
    embedding vector(100),  -- update dim if needed
    cls TEXT
);

CREATE TABLE vectors_100k (
    id BIGSERIAL PRIMARY KEY,
    embedding vector(100),  -- update dim if needed
    cls TEXT
);

CREATE TABLE vectors_200k (
    id BIGSERIAL PRIMARY KEY,
    embedding vector(100),  -- update dim if needed
    cls TEXT
);

CREATE TABLE vectors_500k (
    id BIGSERIAL PRIMARY KEY,
    embedding vector(100),  -- update dim if needed
    cls TEXT
);
EOF

echo "Done."