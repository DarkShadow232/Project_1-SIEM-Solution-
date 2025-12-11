# 🐳 Docker Setup Guide

## Network Traffic Anomaly & Threat Classification

This guide explains how to run the SIEM Dashboard using Docker.

---

## 📋 Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed
- Docker Compose (included with Docker Desktop)

---

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d --build
```

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t network-threat-classifier .

# Run the container
docker run -d -p 5000:5000 --name siem-dashboard network-threat-classifier
```

---

## 🌐 Access the Dashboard

Once running, open your browser and navigate to:

| Page | URL |
|------|-----|
| **Overview** | http://localhost:5000/ |
| **Alerts** | http://localhost:5000/alerts |
| **Outliers** | http://localhost:5000/outliers |
| **Monitor** | http://localhost:5000/monitor |
| **Search** | http://localhost:5000/search |

---

## 🛠️ Development Mode

For development with hot-reload:

```bash
# Start development container
docker-compose --profile dev up siem-dev

# Access at http://localhost:5001
```

---

## 📊 Common Commands

### Container Management

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f siem-dashboard

# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild without cache
docker-compose build --no-cache
```

### Direct Docker Commands

```bash
# List containers
docker ps

# Stop container
docker stop siem-dashboard

# Remove container
docker rm siem-dashboard

# View logs
docker logs siem-dashboard -f

# Execute command in container
docker exec -it siem-dashboard /bin/bash
```

---

## 📁 Volume Mounts

The following directories are mounted for persistent data:

| Host Path | Container Path | Description |
|-----------|----------------|-------------|
| `./project 1/siem_integration` | `/app/siem_integration` | SIEM data files |
| `./project 1/Models` | `/app/Models` | ML models |
| `siem-logs` (volume) | `/app/logs` | Application logs |

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `production` | Flask environment |
| `FLASK_DEBUG` | `0` | Debug mode (0=off, 1=on) |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output |

### Customizing

Edit `docker-compose.yml` to modify:
- Port mappings
- Environment variables
- Volume mounts
- Resource limits

---

## 🏗️ Build Details

The Docker image includes:

- **Python 3.11** (slim base)
- **Gunicorn** WSGI server
- All required ML libraries:
  - scikit-learn
  - pandas
  - numpy
  - xgboost
  - matplotlib
  - plotly
- Flask web framework

---

## 🐛 Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs siem-dashboard

# Verify build
docker-compose build --no-cache
```

### Port already in use

```bash
# Change port in docker-compose.yml or use:
docker-compose up -p 5001:5000
```

### Permission issues

```bash
# On Linux, you may need:
sudo docker-compose up
```

### Memory issues

Add resource limits to `docker-compose.yml`:

```yaml
services:
  siem-dashboard:
    deploy:
      resources:
        limits:
          memory: 2G
```

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🤝 Support

For issues, please open a GitHub issue or contact the development team.

