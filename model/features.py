# ==============================================================================
# MERKMALE FÜR DAS TRAINING – ganzer Datensatz
# ==============================================================================
# Eigentliche Merkmalsberechnung steht in production/merkmale.py (derselbe
# Code wie in API). Hier kommen Schritte dazu, die nur Training braucht:
#   - Länderrisiko (nutzt Label "Is_laundering")
#   - Zahlungsart als 0/1-Spalten (ohne Textspalte)
#
# Start (aus Projektordner, ca. 2 Min.):
#   .venv/bin/python -m model.features
# Liest beide Dateien aus production/data (von etl.py), schreibt model/data/SAML-D_mod.parquet.

import pandas as pd

# Glättung K, Eingabe- und Zieldateien stehen in konfiguration.py
from konfiguration import BASIS, K, MERKMALE_DATEI, STREAM
# Merkmalsberechnung aus Betrieb – so rechnen Training und API gleich
from production.merkmale import meldung, merkmale_berechnen, zahlungsart_spalten


# ==============================================================================
# 1. LADEN
# ==============================================================================

# Lädt alle bereinigten Transaktionen: Vorgeschichte + Stream, chronologisch (von etl.py)
def transaktionen_laden():
    return pd.concat([pd.read_parquet(BASIS), pd.read_parquet(STREAM)], ignore_index=True)


# ==============================================================================
# 2. LÄNDERRISIKO
# Geldwäschequote je Land statt einer festen Länderliste. Länder mit wenigen
# Transaktionen werden zur Gesamtquote gezogen, damit sie keine Zufallswerte
# bekommen. ACHTUNG: nutzt Label -> model.py berechnet es nach Split
# nur aus Trainingsdaten neu, API nimmt Werte aus Modell.
# ==============================================================================

# Berechnet geglättete Geldwäschequote je Land (Ergebnis: Land -> Quote)
def laenderrisiko(df, land_spalte, K=K):
    # Anteil Geldwäsche über alle Transaktionen
    gesamt_quote = df["Is_laundering"].mean()
    # Anzahl Geldwäschefälle und Transaktionen je Land
    stats = df.groupby(land_spalte)["Is_laundering"].agg(["sum", "count"])
    # Geglättete Quote: wenige Transaktionen -> nah an Gesamtquote
    return (stats["sum"] + K * gesamt_quote) / (stats["count"] + K)


# ==============================================================================
# 3. ALLE MERKMALE FÜR TRAINING (auch von manuelle_anomalien.py genutzt)
# ==============================================================================

# Berechnet Merkmale aus merkmale.py, dazu Länderrisiko und Zahlungsart als 0/1-Spalten
def merkmale_fuer_training(df):
    df = merkmale_berechnen(df, melden=True)
    meldung("7. Länderrisiko ...")
    df["Risiko_Land_Empfänger"] = df["Receiver_bank_location"].map(laenderrisiko(df, "Receiver_bank_location"))
    df["Risiko_Land_Sender"] = df["Sender_bank_location"].map(laenderrisiko(df, "Sender_bank_location"))
    # Ersetzt Textspalte Payment_type durch 0/1-Spalten
    meldung("8. Zahlungsart: 0/1-Spalten ...")
    return zahlungsart_spalten(df).drop(columns="Payment_type")


# ==============================================================================
# 4. START ALS SKRIPT
# ==============================================================================

if __name__ == "__main__":
    # Lädt alle bereinigten Transaktionen (Datentypen sind schon richtig) und berechnet Merkmale
    meldung("Lese Transaktionen aus production/data ...")
    df_aml = merkmale_fuer_training(transaktionen_laden())

    # Zeigt alle Spalten. Gespeichert werden ALLE – welche davon Merkmale sind, entscheidet model.py
    meldung(f"Fertig berechnet: {len(df_aml):,} Zeilen".replace(",", ".") + f" und {df_aml.shape[1]} Spalten:")
    for spalte in df_aml.columns:
        print(f"    {spalte}")

    # Speichert alles als SAML-D_mod.parquet (Grundlage für model.py)
    df_aml.to_parquet(MERKMALE_DATEI, index=False)
    meldung(f"Geschrieben: {MERKMALE_DATEI}")


# ==============================================================================
# NOTIZEN
# ==============================================================================
# Spalten vor Feature Engineering (wie im Kaggle-Datensatz, Date + Time = Timestamp):
#   Timestamp, Sender_account, Receiver_account, Amount, Payment_currency, Received_currency,
#   Sender_bank_location, Receiver_bank_location, Payment_type, Is_laundering
#
# Erledigt (siehe model.py):
#   Stark korrelierte Features streichen (|r| > 0,9) – macht Koeffizienten instabil
#   Trainingsdaten selbst splitten
#   Länderrisiko nach Split nur aus Trainingsdaten neu berechnen
#   Label "Is_laundering" vor Training abtrennen, da Zielvariable
