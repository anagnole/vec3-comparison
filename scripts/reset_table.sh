#!/bin/bash
echo "Resetting pgvector test table..."

psql vecdb <<EOF
DROP TABLE IF EXISTS vectors;

CREATE TABLE vectors (
    id BIGSERIAL PRIMARY KEY,
    embedding vector(3),
    cls TEXT
);
EOF

echo "Table reset complete."
