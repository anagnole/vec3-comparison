PostgreSQL 16 installed via Homebrew
pgvector manually compiled/installed using:

wget https://github.com/pgvector/pgvector/archive/refs/tags/v0.8.1.tar.gz
tar -xf v0.8.1.tar.gz
cd pgvector-0.8.1
export PATH="/usr/local/opt/postgresql@16/bin:$PATH"
make
make install

CREATE EXTENSION vector;
