# AML-Projekt – Geldwäscheerkennung mit Stream

Ein LightGBM-Modell bewertet Transaktionen über eine FastAPI in Echtzeit, ein Simulator schickt dafür 
Transaktionen als Stream an die API. Verdächtige Transaktionen erscheinen auf der Monitoring-Seite `/alarme`.

## Projektstruktur

- `model/` – Daten aufbereiten, Merkmale berechnen, Modell trainieren (nur lokal, nicht im Container)
- `production/` – API, Datenbank, Merkmalsberechnung und Simulator, dazu das trainierte Modell in `models/`
- `web/` – Dockerfile und Terraform (Azure)
- `tests/` – pytest-Tests
- `konfiguration.py` – alle einstellbaren Werte

## Voraussetzungen

- Python 3.12
- Docker Desktop (für den Container)
- Azure CLI, Terraform und ein Azure-Konto (nur für die Cloud)

## Installation

Alle Befehle laufen aus dem Projektordner.

Datensatz SAML-D von Kaggle herunterladen und als `model/data/SAML-D.csv` ablegen
(https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml).
Die Daten liegen wegen ihrer Größe nicht im Repository.

```
python3.12 -m venv .venv
.venv/bin/pip install pandas pyarrow numpy scikit-learn lightgbm joblib fastapi uvicorn httpx pytest
```


## Modell trainieren

```
# CSV aufbereiten und aufteilen (ca. 15 s)
.venv/bin/python -m model.etl

# Merkmale berechnen → model/data/SAML-D_mod.parquet (ca. 2 Min.)
.venv/bin/python -m model.features

# Modell trainieren → production/models/lightgbm_<Zeitstempel>.joblib (ca. 5 Min.)
.venv/bin/python -m model.model
```

Ein trainiertes Modell liegt bereits in `production/models/`. Die API lädt immer das neueste.
Ohne neues Training reicht `model.etl`, die Datenbank braucht dessen Dateien.

Für die Cloud zusätzlich `web/terraform.tfvars` mit den eigenen Werten anlegen (nicht ins Repository):

```
subscription_id = "<Azure-Abonnement-ID>"
email_budget    = "<E-Mail für den Budget-Alarm>"
budget_euro     = 5
```


## API lokal starten (Docker)

```
# Datenbank frisch aufbauen
.venv/bin/python -m production.datenbank

# Image bauen
docker build -f web/Dockerfile -t aml-api .

# Container starten (Ctrl+C = Ende)
docker run --rm --name aml-api -p 8000:8000 aml-api

    # 2. Terminal: Stream senden → http://127.0.0.1:8000/alarme
    .venv/bin/python -m production.simulator --anzahl 2000

# Container beenden (aus einem anderen Terminal)
docker stop aml-api

# Docker-Cache löschen (gibt Speicherplatz frei)
docker builder prune -f

# Tests
.venv/bin/python -m pytest -v
```


## API in der Cloud starten (Azure, optional)

Die App läuft in Azure dauerhaft und kostet Geld – danach immer `destroy` ausführen.

```
# Bei Azure anmelden
az login

# Terraform vorbereiten
terraform -chdir=web init

# 1. Leere Registry anlegen
terraform -chdir=web apply -target=azurerm_container_registry.aml -target=azurerm_consumption_budget_resource_group.aml

    # 2a. Datenbank frisch aufbauen, dann Image für Azure-Prozessor bauen
    .venv/bin/python -m production.datenbank
    docker build --platform linux/amd64 -f web/Dockerfile -t $(terraform -chdir=web output -raw registry)/aml-api:latest .

    # 2b. Bei Registry anmelden
    terraform -chdir=web output -raw registry_passwort | docker login $(terraform -chdir=web output -raw registry) -u $(terraform -chdir=web output -raw registry_benutzer) --password-stdin

    # 2c. Image hochladen
    docker push $(terraform -chdir=web output -raw registry)/aml-api:latest

# 3. App starten
terraform -chdir=web apply

# Adresse anzeigen (+ /alarme = Monitoring)
terraform -chdir=web output api_url

# Stream an die Cloud (Adresse wird automatisch eingesetzt)
.venv/bin/python -m production.simulator --url $(terraform -chdir=web output -raw api_url) --anzahl 2000

# Alles löschen, sonst laufen Kosten
terraform -chdir=web destroy
```


## Hinweise

- API-Doku zum Ausprobieren: http://127.0.0.1:8000/docs
- Alle Transaktionen, die an die API gehen, werden in `production/data/aml.db` gespeichert.
  Für einen sauberen Neustart die Datenbank neu aufbauen (`production.datenbank`).
- Einstellungen wie Bargeld-Schwelle oder Zeitraum stehen in `konfiguration.py`.
  Nach einer Änderung die Pipeline neu laufen lassen.
