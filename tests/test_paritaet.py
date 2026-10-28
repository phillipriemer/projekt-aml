# ==============================================================================
# PARITÄTSTEST – rechnet API dieselben Merkmale wie Training?
# ==============================================================================
# API berechnet Merkmale Transaktion für Transaktion aus Vorgeschichte,
# features.py dagegen für alle Transaktionen auf einmal. Beide Wege müssen
# dasselbe ergeben – sonst sieht Modell im Betrieb andere Werte als im Training.
#
# Start (aus Projektordner, ca. 2 Min., braucht echte Daten und Modell):
#   .venv/bin/python -m pytest -m langsam -v

import joblib
import numpy as np
import pandas as pd
import pytest

from konfiguration import BASIS, MERKMALE_DATEI, STREAM
from production.api import modell_datei
from production.datenbank import ROHSPALTEN, datenbank_aufbauen, transaktion_aufnehmen, verbinden

# Anzahl neuer Transaktionen, die verglichen werden
N = 3_000


# Läuft nur mit "-m langsam" (siehe pytest.ini)
@pytest.mark.langsam
def test_api_rechnet_wie_training(tmp_path):
    # Lädt Modell (kennt Merkmalsliste und Länderrisiko)
    artefakt = joblib.load(modell_datei())
    # Baut Datenbank wie im Betrieb in einem Temp-Ordner auf (aml.db bleibt unberührt)
    datei = tmp_path / "test.db"
    anzahl = datenbank_aufbauen(datei, BASIS)
    verbindung = verbinden(datei)

    # Weg 1 (API): nimmt erste N neuen Transaktionen einzeln auf, so wie die API
    neue = pd.read_parquet(STREAM).head(N)[ROHSPALTEN]
    online = pd.concat([transaktion_aufnehmen(verbindung, t, artefakt) for t in neue.to_dict("records")])

    # Weg 2 (Training): dieselben Transaktionen aus SAML-D_mod.parquet
    offline = pd.read_parquet(MERKMALE_DATEI).iloc[anzahl:anzahl + N]
    # Länderrisiko kommt in beiden Wegen aus dem Modell
    quote = artefakt["gesamt_quote"]
    offline["Risiko_Land_Empfänger"] = offline["Receiver_bank_location"].map(artefakt["risiko_land_empfaenger"]).fillna(quote)
    offline["Risiko_Land_Sender"] = offline["Sender_bank_location"].map(artefakt["risiko_land_sender"]).fillna(quote)

    # Vergleicht jedes Merkmal, das Modell nutzt (winzige Toleranz für Rundungen)
    merkmale = artefakt["merkmale"]
    gleich = np.isclose(online[merkmale].astype(float).to_numpy(), offline[merkmale].astype(float).to_numpy())
    abweichend = [m for m, ok in zip(merkmale, gleich.all(axis=0)) if not ok]
    assert not abweichend, f"Diese Merkmale weichen ab: {abweichend}"
