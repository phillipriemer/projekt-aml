# ==============================================================================
# DOCKER-IMAGE DER SCORING-API – Ordner production + konfiguration.py, nichts aus model
# ==============================================================================
# Liegt in web/ (Bereitstellung), baut aber aus Projektordner (Build-Kontext ".").
# Bauen (aus Projektordner, vorher Datenbank frisch aufbauen):
#   .venv/bin/python -m production.datenbank
#   docker build -f web/Dockerfile -t aml-api .
# Starten (danach http://127.0.0.1:8000/alarme), beenden mit Ctrl+C:
#   docker run --rm --name aml-api -p 8000:8000 aml-api
# Für Azure auf einem Mac mit Apple-Chip beim Bauen zusätzlich: --platform linux/amd64


# ==============================================================================
# 1. GRUNDLAGE
# ==============================================================================

# Startet von einem schlanken Linux mit Python 3.12
FROM python:3.12-slim

# Installiert OpenMP-Bibliothek, die LightGBM braucht (und räumt danach auf)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Arbeitsordner im Container
WORKDIR /app


# ==============================================================================
# 2. PAKETE UND CODE
# ==============================================================================

# Installiert Pakete zuerst (ändern sich selten -> Docker verwendet diese Schicht wieder).
# Dieselben Versionen wie lokal in .venv; scikit-learn braucht joblib zum Laden des Modells.
RUN pip install --no-cache-dir fastapi==0.143.0 uvicorn==0.54.0 pandas==3.0.6 numpy==2.5.3 \
    lightgbm==4.7.0 scikit-learn==1.9.1 joblib==1.6.0

# Kopiert zentrale Konfiguration (alle einstellbaren Werte)
COPY konfiguration.py .
# Kopiert Ordner production: Code, Modell und Datenbank (aml.db, ca. 1,7 GB).
# Was nicht hinein soll, steht in Dockerfile.dockerignore.
COPY production/ production/


# ==============================================================================
# 3. START
# ==============================================================================

# API lauscht im Container auf Port 8000
EXPOSE 8000

# Startet API: 0.0.0.0 = von außerhalb des Containers erreichbar,
# ein Worker, weil SQLite nur einen Schreiber erlaubt
CMD ["uvicorn", "production.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
