# ==============================================================================
# TESTS DER SCORING-API – prüfen automatisch, ob die API richtig arbeitet
# ==============================================================================
# Jeder Test schickt Anfragen an die API (ohne laufenden Server) und prüft mit
# "assert", ob die Antwort stimmt. Jeder Test startet mit einer leeren
# Test-Datenbank – production/data/aml.db bleibt unberührt.
#
# Start (aus Projektordner):
#   .venv/bin/python -m pytest -v

import pytest
from fastapi.testclient import TestClient

from production import api
from production.datenbank import leere_datenbank, transaktion_aufnehmen


# ==============================================================================
# HILFEN
# ==============================================================================

# Startet API vor jedem Test mit frischer, leerer Test-Datenbank
@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Legt leere Test-Datenbank in einem Temp-Ordner an
    datei = tmp_path / "test.db"
    leere_datenbank(datei).close()
    # Lässt API diese Datei statt aml.db öffnen
    monkeypatch.setattr(api, "DATENBANK", datei)
    # Startet API (Modell laden, Datenbank öffnen) und gibt sie an Test weiter
    with TestClient(api.app) as c:
        yield c


# Baut eine Transaktion im Format der API
def transaktion(zeit, sender, empfaenger, betrag, typ="ACH", land="UK", waehrung="UK pounds"):
    return {"Timestamp": zeit, "Sender_account": sender, "Receiver_account": empfaenger,
            "Amount": betrag, "Payment_currency": "UK pounds", "Received_currency": waehrung,
            "Sender_bank_location": "UK", "Receiver_bank_location": land, "Payment_type": typ}


# Schickt Demo-Fall: 4 Bareinzahlungen knapp unter 10.000, dann 37.000 in die Emirate
def demo_fall(client):
    for zeit in ["2023-07-10T09:00:00", "2023-07-10T12:00:00", "2023-07-11T09:00:00", "2023-07-11T14:00:00"]:
        client.post("/transaktion", json=transaktion(zeit, 111, 222, 9500, "Cash Deposit"))
    antwort = client.post("/transaktion", json=transaktion(
        "2023-07-11T15:00:00", 222, 333, 37000, "Cross-border", "UAE", "Dirham"))
    return antwort.json()


# ==============================================================================
# TESTS
# ==============================================================================

# 1. API läuft und Modell ist geladen
def test_api_laeuft(client):
    antwort = client.get("/health")
    assert antwort.json() == {"status": "ok", "modell_geladen": True}


# 2. Eine Transaktion wird bewertet und gespeichert
def test_transaktion_wird_bewertet_und_gespeichert(client):
    antwort = client.post("/transaktion", json=transaktion("2023-07-10T10:00:00", 1, 2, 250.0))
    # Antwort ist ok und enthält eine Wahrscheinlichkeit zwischen 0 und 100 %
    assert antwort.status_code == 200
    assert 0 <= antwort.json()["wahrscheinlichkeit_prozent"] <= 100
    # Transaktion und Bewertung stehen in der Datenbank
    db = api.speicher["db"]
    assert db.execute("SELECT COUNT(*) FROM transaktionen").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM vorhersagen").fetchone()[0] == 1


# 3. Ungültige Eingaben werden mit Fehler 422 abgelehnt
def test_ungueltige_eingabe_wird_abgelehnt(client):
    # Negativer Betrag
    assert client.post("/transaktion", json=transaktion("2023-07-10T10:00:00", 1, 2, -50)).status_code == 422
    # Unbekannte Zahlungsart
    assert client.post("/transaktion", json=transaktion("2023-07-10T10:00:00", 1, 2, 50, "Bitcoin")).status_code == 422
    # Fehlendes Feld
    ohne_land = transaktion("2023-07-10T10:00:00", 1, 2, 50)
    del ohne_land["Sender_bank_location"]
    assert client.post("/transaktion", json=ohne_land).status_code == 422


# 4. Merkmale nutzen Vorgeschichte: erste Zahlung an ein Konto ist "neu", zweite nicht mehr
def test_merkmale_nutzen_vorgeschichte(client):
    erste = transaktion_aufnehmen(api.speicher["db"], transaktion("2023-07-10T10:00:00", 5, 6, 100.0), api.artefakt)
    zweite = transaktion_aufnehmen(api.speicher["db"], transaktion("2023-07-10T11:00:00", 5, 6, 100.0), api.artefakt)
    assert erste["Neuer_Sender_Empfänger"].item() == 1
    assert zweite["Neuer_Sender_Empfänger"].item() == 0


# 5. Typischer Geldwäschefall löst einen Alarm aus
def test_demo_fall_loest_alarm_aus(client):
    assert demo_fall(client)["alarm"] is True


# 6. Alarm erscheint auf der Monitoring-Seite; bestätigt ein Analyst ihn, ist Precision 100 %
def test_alarm_und_feedback_auf_monitoring_seite(client):
    ergebnis = demo_fall(client)
    client.post("/feedback", json={"transaktion_id": ergebnis["transaktion_id"], "geldwaesche": True})
    seite = client.get("/alarme").text
    assert "Konto 222 → 333" in seite
    assert "(bestätigt)" in seite
    assert "Precision: 100,0 %" in seite
