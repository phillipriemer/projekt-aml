# ==============================================================================
# ETL – Rohdaten (SAML-D.csv) einmalig laden, bereinigen und für Betrieb aufteilen
# ==============================================================================
# CSV wird nur hier gelesen. Danach arbeiten alle Schritte mit Parquet (klein,
# schnell, merkt sich Datentypen). Spaltennamen bleiben wie im Kaggle-Datensatz –
# nur Date + Time werden zu einem Timestamp zusammengefügt.
#
# Ergebnis: zwei Dateien in production/data, Daten liegen also nur einmal vor:
#   SAML-D_9.4mio.parquet      -> alle außer letzten 100.000 (9,4 Mio.) = Vorgeschichte der API
#   SAML-D_last_0.1mio.parquet -> letzte 100.000 (0,1 Mio.) = "neue" Transaktionen für Simulator
# Training liest beide zusammen (transaktionen_laden in features.py).
#
# Start (aus Projektordner):
#   .venv/bin/python -m model.etl

import pandas as pd

# Pfade und Größe des Streams stehen in konfiguration.py
from konfiguration import BASIS, ROHDATEN, STREAM, STREAM_ANZAHL


# ==============================================================================
# 1. LADEN
# ==============================================================================

# Liest CSV, fügt Date + Time zu Timestamp zusammen und entfernt nicht benötigte Spalten
def daten_laden():
    df = pd.read_csv(ROHDATEN)
    df.insert(0, "Timestamp", pd.to_datetime(df["Date"] + " " + df["Time"]))
    return df.drop(columns=["Date", "Time", "Laundering_type"])



# ==============================================================================
# 2. START ALS SKRIPT
# ==============================================================================

if __name__ == "__main__":
    # Liest CSV einmalig und bereinigt sie
    df = daten_laden()
    # Teilt in Vorgeschichte und Stream und speichert beides als Parquet
    BASIS.parent.mkdir(parents=True, exist_ok=True)
    df.iloc[:-STREAM_ANZAHL].to_parquet(BASIS, index=False)
    df.iloc[-STREAM_ANZAHL:].to_parquet(STREAM, index=False)
    print(f"Geschrieben: {BASIS.name} ({len(df) - STREAM_ANZAHL:,} Zeilen) und "
          f"{STREAM.name} ({STREAM_ANZAHL:,} Zeilen)".replace(",", "."))
