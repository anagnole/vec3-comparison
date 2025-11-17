# Project Scripts Overview

This project includes helper scripts that automate common tasks such as
managing PostgreSQL, resetting pgvector tables, activating the environment,
and running smoke tests.

## Available Scripts

### start_db.sh
Starts PostgreSQL 16.
Command:
./scripts/start_db.sh

### stop_db.sh
Stops PostgreSQL 16.
Command:
./scripts/stop_db.sh

### restart_db.sh
Restarts PostgreSQL service.
Command:
./scripts/restart_db.sh

### reset_table.sh
Drops and recreates the 'vectors' table.
Useful for smoke tests.
Command:
./scripts/reset_table.sh

### smoke_pgvector.sh
Runs the pgvector smoke test.
Command:
./scripts/smoke_pgvector.sh

### smoke_chroma.sh
Runs the Chroma smoke test.

### setup.sh
Initializes venv and installs Python dependencies.

### dev.sh
Activates venv and starts PostgreSQL 16.
Use this when beginning development.
