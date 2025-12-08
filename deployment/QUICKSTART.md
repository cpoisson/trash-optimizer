# Quick Start Guide

## First Time Setup

```bash
# From project root
cd deployment

# Run setup script (creates .env and secrets/ directory)
./setup.sh

# Copy your GCP credentials
cp ~/path/to/your-service-account-key.json secrets/gcp-credentials.json

# Edit .env with your actual API keys
nano .env  # or vim, code, etc.
```

## Build and Run

```bash
# From deployment/ directory
docker-compose up --build
```

Access the app at: http://localhost:8501

## Common Commands

```bash
# Start (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up --build -d

# Check status
docker-compose ps
```

## Using Makefile

Alternatively, use the Makefile for shortcuts:

```bash
make setup      # Initial setup
make build      # Build image
make run        # Start services
make logs       # View logs
make stop       # Stop services
make rebuild    # Rebuild and restart
```

## File Structure

```
deployment/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration config
├── supervisord.conf        # Process manager config
├── .env.template          # Environment variables template
├── .env                   # Your secrets (gitignored)
├── secrets/               # Directory for credential files (gitignored)
│   └── gcp-credentials.json
├── setup.sh               # Automated setup script
├── Makefile               # Command shortcuts
└── README.md              # Full documentation
```

## Secrets Management

- **Never commit** `.env` or `secrets/` to git (already in .gitignore)
- API keys go in `.env`
- GCP JSON credential file goes in `secrets/gcp-credentials.json`
- Both are mounted into the container at runtime

## Troubleshooting

**Port 8501 already in use:**
```bash
# Find what's using the port
lsof -i :8501
# Kill it or change the port in docker-compose.yml
```

**Build fails:**
```bash
# Clean everything and rebuild
docker-compose down -v
docker-compose build --no-cache
```

**Can't connect to inference API:**
- Check logs: `docker-compose logs inference`
- Verify HF_TOKEN and HF_MODEL_REPO_ID in .env

**BigQuery errors:**
- Check GCP credentials file exists: `ls secrets/gcp-credentials.json`
- Verify service account has BigQuery permissions
- Check GCP_PROJECT and GCP_DATASET values in .env

For complete documentation, see [README.md](README.md)
