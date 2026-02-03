# vec3 API

This API allows the React frontend to trigger Python scripts and control Docker.

Quick start (from project root):

```bash
cd api
npm install

cd ..
source .venv/bin/activate

cd api
npm start
```

Endpoints:
- `POST /generate` {size, dim, out} -> runs `python3 vec3/generate_data.py`
- `POST /ingest/:target` {data_dir} -> runs `ingest_chroma.py` or `ingest_pgvector.py`
- `POST /docker/up` -> `docker compose up -d`
- `POST /docker/down` -> `docker compose down`