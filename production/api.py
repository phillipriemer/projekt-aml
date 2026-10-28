# ==============================================================================
# SCORING-API FÜR DIE GELDWÄSCHEERKENNUNG (FastAPI)
# ==============================================================================
# Nimmt Transaktionen entgegen, berechnet ihre Merkmale und gibt
# Geldwäsche-Wahrscheinlichkeit des LightGBM-Modells zurück. Jede Transaktion
# wird in production/data/aml.db gespeichert. Ein- und Ausgabe laufen NUR über
# diese API (Simulator schickt, Seite /alarme zeigt Alarme live).
#
# Endpunkte (ausprobieren unter http://127.0.0.1:8000/docs):
#   POST /transaktion  -> eine Transaktion bewerten
#   POST /feedback     -> echtes Label nachreichen (z. B. vom Analysten)
#   GET  /alarme       -> Monitoring im Browser: Kennzahlen + Alarme
#   GET  /health       -> läuft API?
#
# Start (aus Projektordner), beenden mit Ctrl+C:
#   .venv/bin/python -m production.api

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Adresse, Pfade, Drift-Grenzen und Zahlungsarten stehen in konfiguration.py
from konfiguration import (DATENBANK, HOST, MODELLE, NEU_LADEN_S, PORT, PSI_DRIFT, PSI_STABIL,
                           REFERENZ_TAGE, ZAHLUNGSARTEN)
from production.datenbank import LOCK, transaktion_aufnehmen, verbinden


# ==============================================================================
# ZUSTAND DER API
# ==============================================================================

# Schreibt Fehler ins Terminal, ohne API abstürzen zu lassen
log = logging.getLogger("aml_api")

# Hier liegt nach Start Modell samt allem, was beim Training festgelegt wurde
artefakt = {}
# Hier liegen nach Start Datenbankverbindung und Drift-Referenz
speicher = {}


# ==============================================================================
# 1. START UND ENDE DER API
# Alles vor "yield" läuft einmal beim Start, alles danach beim Beenden
# ==============================================================================

# Findet neuestes Modell in production/models (Name enthält Datum)
def modell_datei():
    return max(MODELLE.glob("lightgbm_20*.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lädt Modell mit Merkmalsliste, Schwelle und Länderrisiko
    artefakt.update(joblib.load(modell_datei()))
    # Bricht ab, wenn Datenbank noch nicht aufgebaut wurde
    datei = DATENBANK
    if not datei.exists():
        raise RuntimeError(f"{datei} fehlt - erst '.venv/bin/python -m production.datenbank' ausführen")
    # Öffnet Datenbank
    speicher["db"] = verbinden(datei)
    # Lädt einmalig Vergleichsbeträge für Drift-Messung (dauert im Container 1-2 Min.)
    speicher["referenz"] = referenz_betraege(speicher["db"])
    print(f"API bereit: Modell {artefakt['version']}, Datenbank {datei.name} - http://{HOST}:{PORT}/docs")
    yield
    # Räumt beim Beenden auf
    speicher["db"].close()
    artefakt.clear()
    speicher.clear()


# Erstellt API (Titel erscheint oben auf Seite /docs)
app = FastAPI(title="AML-Scoring", lifespan=lifespan)


# ==============================================================================
# 2. EINGABE UND AUSGABE
# Passt eine Anfrage nicht (Feld fehlt, Betrag <= 0, unbekannte Zahlungsart),
# lehnt FastAPI sie automatisch mit Fehler 422 ab
# ==============================================================================

# So muss eine Transaktion aussehen, die an API geschickt wird.
# Feldnamen wie im Kaggle-Datensatz SAML-D – dieselben wie in Daten und Datenbank
class Transaktion(BaseModel):
    # Beispiel, das auf Seite /docs vorausgefüllt ist
    model_config = {"json_schema_extra": {"examples": [{
        "Timestamp": "2023-08-22T23:00:00", "Sender_account": 8724731955, "Receiver_account": 2769355426,
        "Amount": 1459.15, "Payment_currency": "UK pounds", "Received_currency": "UK pounds",
        "Sender_bank_location": "UK", "Receiver_bank_location": "UK", "Payment_type": "Cash Deposit"}]}}

    Timestamp: datetime
    Sender_account: int
    Receiver_account: int
    # Betrag muss größer als 0 sein
    Amount: float = Field(gt=0)
    Payment_currency: str
    Received_currency: str
    Sender_bank_location: str
    Receiver_bank_location: str
    # Nur Zahlungsarten aus konfiguration.py sind erlaubt
    Payment_type: Literal[tuple(ZAHLUNGSARTEN)]


# So sieht Antwort der API aus
class Bewertung(BaseModel):
    # Geldwäsche-Wahrscheinlichkeit in Prozent
    wahrscheinlichkeit_prozent: float
    # true, wenn Schwelle erreicht ist
    alarm: bool
    # Alarmschwelle aus Training, in Prozent
    schwelle_prozent: float
    modellversion: str
    # Nummer in Datenbank, z. B. für POST /feedback
    transaktion_id: int


# So sieht ein nachgereichtes echtes Label aus (Analyst hat Alarm geprüft)
class Feedback(BaseModel):
    transaktion_id: int
    geldwaesche: bool


# ==============================================================================
# 3. BEWERTEN
# ==============================================================================

# Berechnet Merkmale, fragt Modell und protokolliert Bewertung
def bewerten(t: Transaktion):
    # Startet Stoppuhr für Latenz
    t0 = time.perf_counter()
    # Berechnet Merkmale und speichert Transaktion (datenbank.py) – Feldnamen passen direkt
    zeile = transaktion_aufnehmen(speicher["db"], t.model_dump(), artefakt)
    # Fragt Modell nach Geldwäsche-Wahrscheinlichkeit (Merkmale per Name, gleiche Reihenfolge wie im Training)
    p = float(artefakt["modell"].predict_proba(zeile[artefakt["merkmale"]].astype(float))[:, 1][0])
    transaktion_id = int(zeile.index[0])
    # Alarm, wenn Wahrscheinlichkeit Schwelle aus Training erreicht
    alarm = p >= artefakt["schwelle"]
    # Protokolliert Bewertung für Monitoring
    with LOCK:
        speicher["db"].execute(
            "INSERT INTO vorhersagen (transaktion_id, zeit, wahrscheinlichkeit, alarm, modellversion, latenz_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (transaktion_id, f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}", p, int(alarm), artefakt["version"],
             1000 * (time.perf_counter() - t0)))
        speicher["db"].commit()
    # Gibt Antwort in Prozent zurück
    return Bewertung(wahrscheinlichkeit_prozent=round(100 * p, 4), alarm=alarm,
                     schwelle_prozent=round(100 * artefakt["schwelle"], 4),
                     modellversion=artefakt["version"], transaktion_id=transaktion_id)


# ==============================================================================
# 4. ENDPUNKTE
# "def" statt "async def": LightGBM rechnet (CPU), es wartet auf nichts
# ==============================================================================

# POST /transaktion: bewertet eine Transaktion
@app.post("/transaktion", response_model=Bewertung)
def transaktion(t: Transaktion):
    try:
        return bewerten(t)
    except Exception:
        # Schreibt Fehler ins Terminal und antwortet mit Fehler 500
        log.exception("Bewertung fehlgeschlagen: %s", t)
        raise HTTPException(status_code=500, detail="Bewertung fehlgeschlagen")


# POST /feedback: speichert echtes Label (kommt im Betrieb erst später)
@app.post("/feedback")
def feedback(f: Feedback):
    with LOCK:
        db = speicher["db"]
        # Antwortet mit 404, wenn Transaktion nicht existiert
        if db.execute("SELECT 1 FROM transaktionen WHERE rowid = ?", (f.transaktion_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail=f"Transaktion {f.transaktion_id} unbekannt")
        # Speichert Label
        db.execute("INSERT INTO feedback (transaktion_id, zeit, geldwaesche) VALUES (?, ?, ?)",
                   (f.transaktion_id, f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}", int(f.geldwaesche)))
        db.commit()
    return {"gespeichert": True}


# Schreibt eine Zahl im deutschen Format, z. B. 9.500,00
def zahl(wert, stellen=2):
    return f"{wert:,.{stellen}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# GET /alarme: Monitoring-Seite – Kennzahlen oben, darunter Alarme (neueste oben)
@app.get("/alarme", response_class=HTMLResponse)
def alarme(anzahl: int = Query(50, ge=1, le=500)):
    # Berechnet Kennzahlen aus allen bisherigen Bewertungen
    k = kennzahlen(bewertungen_laden(speicher["db"]), speicher["referenz"])

    # Formatiert einen Wert; fehlende Werte erscheinen als "–"
    def wert(x, stellen=1, faktor=1, einheit=""):
        return "–" if x is None else f"{zahl(x * faktor, stellen)}{einheit}"

    # Baut Kopfzeile mit Kennzahlen
    kopf = (f"Bewertet: {zahl(k['bewertet'], 0)} · Alarme: {k['alarme']} · Latenz: {wert(k['latenz_median_ms'], einheit=' ms')} · "
            f"Drift: {k['drift'] or '–'} (PSI {wert(k['psi_betrag'], 3)}) · "
            f"Precision: {wert(k['precision'], faktor=100, einheit=' %')}")
    # Baut eine Textzeile je Alarm
    zeilen = [f"⚠ {z.zeitpunkt}  Konto {z.sender} → {z.empfaenger}  {zahl(z.betrag)}  {z.zahlungsart}  "
              f"{zahl(z.wahrscheinlichkeit_prozent, 1)} %  ({z.feedback})"
              for z in alarme_laden(speicher["db"], anzahl).itertuples()]
    # Gibt eine schlichte HTML-Seite zurück, die sich selbst neu lädt
    return (f'<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="{NEU_LADEN_S}"><title>AML-Alarme</title></head>'
            f'<body style="font-family:monospace;margin:24px"><h3>AML-Monitoring</h3><p>{kopf}</p>'
            f'<pre>{chr(10).join(zeilen) or "Noch keine Alarme."}</pre></body></html>')


# GET /health: zeigt, ob API läuft und Modell geladen ist
@app.get("/health")
def health():
    return {"status": "ok", "modell_geladen": "modell" in artefakt}


# ==============================================================================
# 5. MONITORING: DATEN AUS DER DATENBANK
# Service -> Bewertungen, Latenz | Daten -> Drift der Beträge |
# Modell  -> Alarme, Precision aus Feedback der Analysten
# ==============================================================================

# Lädt alle Bewertungen mit Betrag und (falls vorhanden) letztem Feedback
def bewertungen_laden(verbindung):
    with LOCK:
        return pd.read_sql_query(
            """
            SELECT v.alarm, v.latenz_ms, t."Amount", f.geldwaesche AS label
            FROM vorhersagen v
            LEFT JOIN transaktionen t ON t.rowid = v.transaktion_id
            LEFT JOIN feedback f ON f.id = (SELECT MAX(id) FROM feedback WHERE transaktion_id = v.transaktion_id)
            """, verbindung)


# Lädt Beträge der letzten 30 Tage VOR neuen Transaktionen (Referenz für Drift).
# Alte Zeilen erkennt man daran, dass sie ein Label haben.
def referenz_betraege(verbindung):
    with LOCK:
        return pd.read_sql_query(
            """
            SELECT "Amount" FROM transaktionen
            WHERE "Is_laundering" IS NOT NULL
              AND "Timestamp" >= (SELECT datetime(MAX("Timestamp"), ?) FROM transaktionen
                                  WHERE "Is_laundering" IS NOT NULL)
            """, verbindung, params=(f"-{REFERENZ_TAGE} days",))["Amount"].to_numpy()


# Lädt letzte Alarme (neueste zuerst) mit Transaktion und Feedback-Status
def alarme_laden(verbindung, anzahl):
    with LOCK:
        return pd.read_sql_query(
            """
            SELECT v.transaktion_id AS transaktion, t."Timestamp" AS zeitpunkt,
                   t."Sender_account" AS sender, t."Receiver_account" AS empfaenger,
                   t."Amount" AS betrag, t."Payment_type" AS zahlungsart,
                   ROUND(100 * v.wahrscheinlichkeit, 2) AS wahrscheinlichkeit_prozent,
                   CASE f.geldwaesche WHEN 1 THEN 'bestätigt' WHEN 0 THEN 'Fehlalarm' ELSE 'offen' END AS feedback
            FROM vorhersagen v
            JOIN transaktionen t ON t.rowid = v.transaktion_id
            LEFT JOIN feedback f ON f.id = (SELECT MAX(id) FROM feedback WHERE transaktion_id = v.transaktion_id)
            WHERE v.alarm = 1
            ORDER BY v.id DESC
            LIMIT ?
            """, verbindung, params=(anzahl,))


# ==============================================================================
# 6. MONITORING: KENNZAHLEN
# ==============================================================================

# PSI (Population Stability Index): Wie stark weicht Verteilung der neuen
# Beträge von Referenz ab? Vergleicht Anteile in 10 Klassen.
def psi(referenz, werte):
    # Ohne Daten keine Aussage
    if len(referenz) == 0 or len(werte) == 0:
        return None
    # Teilt Referenz in 10 gleich große Klassen (Dezile)
    grenzen = np.unique(np.quantile(referenz, np.linspace(0, 1, 11)[1:-1]))
    # Anteil der Beträge je Klasse, für Referenz und neue Werte
    anteile = [np.bincount(np.searchsorted(grenzen, x, side="right"), minlength=len(grenzen) + 1) / len(x)
               for x in (referenz, werte)]
    # Verhindert ln(0) bei leeren Klassen
    ref, neu = (np.clip(a, 1e-4, None) for a in anteile)
    # PSI = Summe (neu - ref) * ln(neu / ref)
    return round(float(np.sum((neu - ref) * np.log(neu / ref))), 4)


# Berechnet alle Kennzahlen für Kopfzeile der Seite /alarme
def kennzahlen(df, referenz):
    # Alarme, die ein Analyst schon geprüft hat (Feedback vorhanden)
    analyst = df[df["label"].notna()]
    # Precision = Anteil der geprüften Alarme, die wirklich Geldwäsche waren
    precision = analyst["label"].mean() if len(analyst) else None

    # Misst Drift der Beträge
    drift = psi(referenz, df["Amount"].dropna().to_numpy())
    # Fasst alles zusammen (fehlende Werte = None)
    return {
        "bewertet": len(df),
        "alarme": int(df["alarm"].sum()),
        "latenz_median_ms": round(float(df["latenz_ms"].median()), 1) if len(df) else None,
        "psi_betrag": drift,
        "drift": None if drift is None else ("stabil" if drift < PSI_STABIL else
                                             "beobachten" if drift < PSI_DRIFT else "Drift"),
        "precision": None if precision is None else round(float(precision), 3),
    }


# ==============================================================================
# 7. START ALS SKRIPT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn

    # Startet Webserver mit einem Worker (SQLite erlaubt nur einen Schreiber)
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
