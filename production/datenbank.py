# ==============================================================================
# DATENBANK DER API – aufbauen, Vorgeschichte abfragen, neue Transaktionen speichern
# ==============================================================================
# production/data/aml.db enthält alle Transaktionen als Rohdaten, außer letzten
# 100.000 – die schickt später der Simulator als "neue" Transaktionen.
# Gespeichert werden nur Rohspalten: Merkmale berechnet merkmale.py bei Bedarf
# aus Vorgeschichte (jederzeit reproduzierbar).
#
# Warum eine Vorgeschichte? Merkmale wie Z-Score oder Fan-out entstehen erst aus
# früheren Transaktionen eines Kontos. Allein wären sie immer 0.
#
# Datenbank aufbauen bzw. zurücksetzen (ca. 1 Min., aus Projektordner):
#   .venv/bin/python -m production.datenbank

import sqlite3
import threading
from pathlib import Path

import pandas as pd

# Dateipfade und längstes Zeitfenster stehen in konfiguration.py
from konfiguration import BASIS, DATENBANK, LAENGSTES_FENSTER
from production.merkmale import meldung, merkmale_berechnen, zahlungsart_spalten


# ==============================================================================
# 1. SPALTEN UND TABELLEN
# Spaltennamen wie im Kaggle-Datensatz SAML-D – in Daten, Datenbank und API gleich
# ==============================================================================

# Felder einer Transaktion, so wie sie in API ankommt (Label fehlt)
ROHSPALTEN = ["Timestamp", "Sender_account", "Receiver_account", "Amount", "Payment_currency",
              "Received_currency", "Sender_bank_location", "Receiver_bank_location", "Payment_type"]

# Zeitformat in Datenbank (Text, lässt sich richtig sortieren)
ZEITFORMAT = "%Y-%m-%d %H:%M:%S"
# So weit wird zurückgeschaut: längstes Merkmalsfenster + 1 Tag Reserve
FENSTER = pd.Timedelta(LAENGSTES_FENSTER) + pd.Timedelta("1D")

# Tabelle "transaktionen": Rohspalten + Label (bei neuen Transaktionen leer)
TABELLE = """
CREATE TABLE IF NOT EXISTS transaktionen (
    "Timestamp" TEXT, "Sender_account" INTEGER, "Receiver_account" INTEGER, "Amount" REAL,
    "Payment_currency" TEXT, "Received_currency" TEXT,
    "Sender_bank_location" TEXT, "Receiver_bank_location" TEXT, "Payment_type" TEXT,
    "Is_laundering" INTEGER
);
"""

# Tabellen für Monitoring (transaktion_id = Zeilennummer in "transaktionen")
LOG_TABELLEN = """
CREATE TABLE IF NOT EXISTS vorhersagen (
    id INTEGER PRIMARY KEY, transaktion_id INTEGER, zeit TEXT,
    wahrscheinlichkeit REAL, alarm INTEGER, modellversion TEXT, latenz_ms REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY, transaktion_id INTEGER, zeit TEXT,
    geldwaesche INTEGER       -- echtes Label nach Prüfung durch Analyst: 1 = Geldwäsche, 0 = keine
);
CREATE INDEX IF NOT EXISTS idx_feedback_transaktion ON feedback (transaktion_id);
"""

# Indizes = Inhaltsverzeichnis, damit Suche nach Konto und Zeit schnell geht
INDIZES = """
CREATE INDEX IF NOT EXISTS idx_sender_zeit     ON transaktionen ("Sender_account", "Timestamp");
CREATE INDEX IF NOT EXISTS idx_empfaenger_zeit ON transaktionen ("Receiver_account", "Timestamp");
CREATE INDEX IF NOT EXISTS idx_paar            ON transaktionen ("Sender_account", "Receiver_account");
"""

# Holt alle früheren Zeilen, von denen Merkmale einer neuen Transaktion abhängen:
#   Sender war Sender       -> Z_Sender, Structuring, Fan-out
#   Empfänger war Empfänger -> Z_Empfänger, Fan-in
#   Sender war Empfänger    -> Bareinzahlung auf Senderkonto
# UNION statt OR, damit SQLite für jeden Teil seinen Index nutzt
SPALTEN_SQL = ", ".join(f'"{s}"' for s in ROHSPALTEN)
VORGESCHICHTE = f"""
SELECT rowid AS id, {SPALTEN_SQL} FROM transaktionen WHERE "Sender_account" = :sender AND "Timestamp" >= :ab
UNION
SELECT rowid AS id, {SPALTEN_SQL} FROM transaktionen WHERE "Receiver_account" = :empfaenger AND "Timestamp" >= :ab
UNION
SELECT rowid AS id, {SPALTEN_SQL} FROM transaktionen WHERE "Receiver_account" = :sender AND "Timestamp" >= :ab
ORDER BY id
"""

# Sperre: Es wird immer nur EINE Transaktion gleichzeitig aufgenommen
LOCK = threading.Lock()


# ==============================================================================
# 2. DATENBANK ÖFFNEN UND AUFBAUEN
# ==============================================================================

# Öffnet Datenbank für API
def verbinden(datei):
    # check_same_thread=False: mehrere Anfragen dürfen dieselbe Verbindung nutzen
    verbindung = sqlite3.connect(datei, check_same_thread=False)
    # WAL-Modus: Lesen und Schreiben blockieren sich nicht gegenseitig
    verbindung.execute("PRAGMA journal_mode=WAL")
    return verbindung


# Legt eine leere Datenbank mit allen Tabellen an (für Tests)
def leere_datenbank(datei):
    verbindung = verbinden(datei)
    verbindung.executescript(TABELLE + LOG_TABELLEN + INDIZES)
    return verbindung


# Baut Datenbank `ziel` aus Parquet-Datei `quelle` auf, gibt Zeilenzahl zurück
def datenbank_aufbauen(ziel, quelle):
    # Löscht eine alte Datenbank samt Hilfsdateien
    ziel = Path(ziel)
    for alt in (ziel, Path(f"{ziel}-wal"), Path(f"{ziel}-shm")):
        alt.unlink(missing_ok=True)

    # Lädt alle Transaktionen und speichert Zeit als Text (lässt sich in SQLite sortieren und vergleichen)
    meldung(f"Lade {Path(quelle).name} ...")
    df = pd.read_parquet(quelle)
    df["Timestamp"] = df["Timestamp"].dt.strftime(ZEITFORMAT)

    verbindung = sqlite3.connect(ziel)
    # Schaltet Sicherheitsprotokolle beim Aufbau ab -> deutlich schneller
    verbindung.execute("PRAGMA journal_mode=OFF")
    verbindung.execute("PRAGMA synchronous=OFF")
    # Legt Tabelle an und schreibt alle Zeilen hinein
    meldung(f"Schreibe {len(df):,} Transaktionen in die Datenbank ...".replace(",", "."))
    verbindung.executescript(TABELLE)
    df.to_sql("transaktionen", verbindung, if_exists="append", index=False, chunksize=100_000)

    # Legt Indizes und Monitoring-Tabellen an
    meldung("Lege Indizes an ...")
    verbindung.executescript(INDIZES + LOG_TABELLEN)
    verbindung.commit()
    # Zählt Transaktionen zur Kontrolle
    anzahl = verbindung.execute("SELECT COUNT(*) FROM transaktionen").fetchone()[0]
    verbindung.close()
    return anzahl


# ==============================================================================
# 3. EINE NEUE TRANSAKTION AUFNEHMEN
# ==============================================================================

# Holt frühere Transaktionen der beteiligten Konten (letzte 31 Tage)
def vorgeschichte_laden(verbindung, sender, empfaenger, zeitpunkt):
    # Frühester Zeitpunkt, der noch gebraucht wird
    ab = (pd.Timestamp(zeitpunkt) - FENSTER).strftime(ZEITFORMAT)
    # Führt Abfrage aus; Index = Zeilennummer in Datenbank
    df = pd.read_sql_query(VORGESCHICHTE, verbindung, index_col="id",
                           params={"sender": int(sender), "empfaenger": int(empfaenger), "ab": ab})
    # Wandelt Text wieder in echte Zeitstempel um
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


# Prüft, ob dieser Sender diesem Empfänger schon einmal Geld geschickt hat (ganze Historie)
def paar_bekannt(verbindung, sender, empfaenger):
    zeile = verbindung.execute('SELECT 1 FROM transaktionen WHERE "Sender_account" = ? AND "Receiver_account" = ? LIMIT 1',
                               (int(sender), int(empfaenger))).fetchone()
    return zeile is not None


# Berechnet alle Merkmale einer neuen Transaktion, speichert sie und gibt Merkmale zurück
# (transaktion = dict mit ROHSPALTEN, artefakt = geladenes Modell aus model.py)
def transaktion_aufnehmen(verbindung, transaktion, artefakt):
    sender, empfaenger = transaktion["Sender_account"], transaktion["Receiver_account"]
    # Schneidet Sekundenbruchteile ab (wie in Trainingsdaten)
    zeitpunkt = pd.Timestamp(transaktion["Timestamp"]).floor("s")

    # Nur eine Transaktion gleichzeitig, damit Vorgeschichte stimmt
    with LOCK:
        # Prüft, ob es Verbindung Sender -> Empfänger schon gab
        neues_paar = not paar_bekannt(verbindung, sender, empfaenger)
        # Holt Vorgeschichte der beteiligten Konten
        vorgeschichte = vorgeschichte_laden(verbindung, sender, empfaenger, zeitpunkt)
        # Gibt neuer Transaktion höchste Nummer (= zeitlich zuletzt)
        neu_id = int(vorgeschichte.index.max()) + 1 if len(vorgeschichte) else 0
        neu = pd.DataFrame([{**transaktion, "Timestamp": zeitpunkt}], index=[neu_id])
        # Hängt neue Transaktion an Vorgeschichte an
        alle = pd.concat([vorgeschichte, neu]) if len(vorgeschichte) else neu

        # Berechnet Merkmale mit Code aus merkmale.py und behält nur neue Zeile
        zeile = zahlungsart_spalten(merkmale_berechnen(alle).loc[[neu_id]])
        # Überschreibt "neues Paar" mit ganzer Historie (Vorgeschichte reicht nur 30 Tage)
        zeile["Neuer_Sender_Empfänger"] = int(neues_paar)
        # Nimmt Länderrisiko aus Modell (unbekanntes Land -> Gesamtquote)
        gesamt = artefakt["gesamt_quote"]
        zeile["Risiko_Land_Empfänger"] = zeile["Receiver_bank_location"].map(artefakt["risiko_land_empfaenger"]).fillna(gesamt)
        zeile["Risiko_Land_Sender"] = zeile["Sender_bank_location"].map(artefakt["risiko_land_sender"]).fillna(gesamt)

        # Speichert nur Rohspalten (Zeit als Text, numpy-Werte als normale Python-Werte)
        werte = [zeitpunkt.strftime(ZEITFORMAT)] + [
            w.item() if hasattr(w, "item") else w for w in (transaktion[s] for s in ROHSPALTEN[1:])]
        cursor = verbindung.execute(
            f"INSERT INTO transaktionen ({SPALTEN_SQL}) VALUES ({', '.join('?' * len(ROHSPALTEN))})", werte)
        verbindung.commit()

    # Gibt Merkmale mit Nummer der Transaktion in Datenbank zurück
    zeile.index = [cursor.lastrowid]
    return zeile


# ==============================================================================
# 4. START ALS SKRIPT
# ==============================================================================

if __name__ == "__main__":
    quelle, ziel = BASIS, DATENBANK
    # Bricht ab, wenn Vorgeschichte noch nicht erzeugt wurde
    if not quelle.exists():
        raise SystemExit(f"{quelle} fehlt - erst model/etl.py ausführen "
                         "(.venv/bin/python -m model.etl)")
    anzahl = datenbank_aufbauen(ziel, quelle)
    meldung(f"Fertig: {ziel} mit {anzahl:,} Transaktionen.".replace(",", "."))
