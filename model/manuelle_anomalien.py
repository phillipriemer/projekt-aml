# ==============================================================================
# MANUELLE ANOMALIEN – künstliche Geldwäschefälle für einen Vergleichslauf (optional)
# ==============================================================================
# Aufgabe: "Füge manuell Anomalien oder potenzielle Geldwäschefälle hinzu, um Dein
# Modell zu trainieren." Dieses Skript erzeugt zwei typische Muster als ganz
# normale Transaktionen – nur im Trainingszeitraum:
#   Structuring -> 4 Bareinzahlungen knapp unter T, danach Überweisung ins Ausland
#   Fan-out     -> ein Konto verteilt Geld in wenigen Stunden an viele neue Konten
#
# Ergebnis ist eine EIGENE Datei. Normale Dateien, API und Betriebsmodell bleiben
# unberührt. Dieses Skript kann gelöscht werden – alles andere läuft weiter.
#
# Start (aus Projektordner), danach Vergleichslauf ohne Speichern des Modells:
#   .venv/bin/python -m model.manuelle_anomalien
#   .venv/bin/python -m model.model --mit-anomalien

import numpy as np
import pandas as pd

# Zieldatei, Bargeld-Schwelle und Einlaufphase stehen in konfiguration.py
from konfiguration import ANOMALIEN_DATEI, EINLAUF_TAGE, T
from model.features import merkmale_fuer_training, transaktionen_laden
from production.merkmale import meldung


# ==============================================================================
# PARAMETER
# ==============================================================================

# So viele Fälle je Muster
ANZAHL_JE_MUSTER = 200
# Künstliche Konten beginnen hier (echte Konten in SAML-D sind höchstens 10-stellig)
ERSTES_KONTO = 20_000_000_000
# Fälle liegen nach Einlaufphase (EINLAUF_TAGE aus konfiguration.py) ...
# ... und vor 60 % der Daten – sicher im Trainingszeitraum (Training endet bei ca. 71 %)
BIS_ANTEIL = 0.6
# Zielländer der Auslandsüberweisungen mit ihrer Währung (in SAML-D häufig bei Geldwäsche)
AUSLAND = {"UAE": "Dirham", "Nigeria": "Naira", "Morocco": "Moroccan dirham", "Albania": "Albanian lek"}

# Zufallsgenerator mit festem Startwert -> bei jedem Lauf dieselben Fälle
zufall = np.random.default_rng(42)


# ==============================================================================
# 1. MUSTER ALS TRANSAKTIONEN
# ==============================================================================

# Baut eine Transaktion mit denselben Spalten wie echte Transaktionen (Label = Geldwäsche)
def transaktion(zeit, sender, empfaenger, betrag, typ, land="UK", waehrung="UK pounds"):
    return {"Timestamp": zeit, "Sender_account": sender, "Receiver_account": empfaenger,
            "Amount": round(float(betrag), 2), "Payment_currency": "UK pounds", "Received_currency": waehrung,
            "Sender_bank_location": "UK", "Receiver_bank_location": land,
            "Payment_type": typ, "Is_laundering": 1}


# Structuring: 4 Bareinzahlungen knapp unter T auf ein Konto, das danach fast alles ins Ausland schickt
def structuring_fall(konto, start):
    einzahler, kunde, ausland = konto, konto + 1, konto + 2
    land = zufall.choice(list(AUSLAND))
    zeilen, zeit = [], start
    for _ in range(4):
        # Nächste Einzahlung 3 bis 30 Stunden später, Betrag zwischen 90 % und 99 % von T
        zeit += pd.Timedelta(hours=float(zufall.uniform(3, 30)))
        zeilen.append(transaktion(zeit, einzahler, kunde, zufall.uniform(0.9 * T, 0.99 * T), "Cash Deposit"))
    # Kurz danach: ca. 95 % der Summe gehen ins Ausland
    summe = sum(z["Amount"] for z in zeilen)
    zeit += pd.Timedelta(hours=float(zufall.uniform(1, 20)))
    zeilen.append(transaktion(zeit, kunde, ausland, 0.95 * summe, "Cross-border", land, AUSLAND[land]))
    return zeilen


# Fan-out: ein Konto verteilt Geld innerhalb von 24 Stunden an 8 bis 15 neue Konten
def fan_out_fall(konto, start):
    zeilen = []
    for i in range(int(zufall.integers(8, 16))):
        zeit = start + pd.Timedelta(hours=float(zufall.uniform(0, 24)))
        zeilen.append(transaktion(zeit, konto, konto + 1 + i, zufall.uniform(500, 5000), "ACH"))
    return zeilen


# ==============================================================================
# 2. START ALS SKRIPT
# ==============================================================================

if __name__ == "__main__":
    # Lädt alle bereinigten Transaktionen (von etl.py)
    meldung("Lese Transaktionen aus production/data ...")
    df = transaktionen_laden()

    # Zeitraum, in dem künstliche Fälle liegen dürfen (nur Training)
    von = df["Timestamp"].min() + pd.Timedelta(days=EINLAUF_TAGE)
    bis = df["Timestamp"].iloc[int(len(df) * BIS_ANTEIL)]
    meldung(f"Erzeuge je {ANZAHL_JE_MUSTER} Fälle Structuring und Fan-out zwischen {von:%d.%m.%Y} und {bis:%d.%m.%Y} ...")

    # Erzeugt alle Fälle; jeder Fall bekommt eigene, neue Kontonummern (Abstand 100)
    zeilen = []
    for nummer in range(2 * ANZAHL_JE_MUSTER):
        start = von + (bis - von) * float(zufall.uniform(0, 1))
        konto = ERSTES_KONTO + 100 * nummer
        zeilen += structuring_fall(konto, start) if nummer < ANZAHL_JE_MUSTER else fan_out_fall(konto, start)
    kuenstlich = pd.DataFrame(zeilen)
    meldung(f"{len(kuenstlich)} künstliche Transaktionen erzeugt")

    # Markiert künstliche Zeilen, damit model.py sie erkennt
    df["Künstlich"] = 0
    kuenstlich["Künstlich"] = 1
    # Sortiert künstliche Zeilen zeitlich ein; "stable" lässt echte Zeilen in ihrer Reihenfolge
    alle = pd.concat([df, kuenstlich[df.columns]], ignore_index=True)
    alle = alle.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    del df

    # Berechnet Merkmale mit derselben Funktion wie features.py (und demselben Code wie API)
    alle = merkmale_fuer_training(alle)

    # Speichert alles als eigene Datei (normale SAML-D_mod.parquet bleibt unverändert)
    alle.to_parquet(ANOMALIEN_DATEI, index=False)
    meldung(f"Geschrieben: {ANOMALIEN_DATEI.name} – weiter mit: .venv/bin/python -m model.model --mit-anomalien")
