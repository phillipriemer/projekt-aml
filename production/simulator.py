# ==============================================================================
# STREAM-SIMULATOR – schickt Transaktionen nacheinander an Scoring-API
# ==============================================================================
# Liest letzte 100.000 Transaktionen von SAML-D (production/data/
# SAML-D_last_0.1mio.parquet) und schickt sie einzeln an POST /transaktion –
# so, als kämen sie live aus Zahlungsverkehr einer Bank.
#
# Jeder Alarm erscheint sofort als eigene Zeile (Warnsignal). Echtes Label
# bekommt API NICHT – es wird nur als Feedback nachgereicht, wie ein
# Analyst es nach Prüfung eines Alarms tun würde.
#
# Start (aus Projektordner, API muss laufen), abbrechen mit Ctrl+C:
#   .venv/bin/python -m production.simulator                         # alle 100.000
#   .venv/bin/python -m production.simulator --anzahl 2000 --rate 20  # kurze Demo

import argparse
import time

import httpx
import pandas as pd

# Adresse der API und Stream-Datei stehen in konfiguration.py
from konfiguration import HOST, PORT, STREAM
from production.datenbank import ROHSPALTEN


# Macht aus einer Zeile Anfrage für POST /transaktion: dieselben Spaltennamen, ohne Label
def anfrage(zeile):
    daten = {spalte: zeile[spalte] for spalte in ROHSPALTEN}
    # Schreibt Zeitstempel als Text (JSON kennt kein Datum)
    daten["Timestamp"] = f"{zeile['Timestamp']:%Y-%m-%dT%H:%M:%S}"
    return daten


def main():
    # ==========================================================================
    # 1. EINSTELLUNGEN AUS DER KOMMANDOZEILE
    # ==========================================================================

    p = argparse.ArgumentParser(description="Schickt SAML-D-Transaktionen an die Scoring-API")
    # Wie viele Transaktionen? (ohne Angabe: alle)
    p.add_argument("--anzahl", type=int, default=None, help="so viele Transaktionen senden (ohne Angabe: alle)")
    # Wie schnell? (0 = so schnell wie möglich)
    p.add_argument("--rate", type=float, default=0, help="Transaktionen pro Sekunde (0 = so schnell wie möglich)")
    # Wohin? (lokal oder Adresse in Cloud)
    p.add_argument("--url", default=f"http://{HOST}:{PORT}", help="Adresse der API (Cloud: --url <api_url>)")
    args = p.parse_args()

    # ==========================================================================
    # 2. VERBINDUNG PRÜFEN UND DATEN LADEN
    # ==========================================================================

    # Öffnet eine Verbindung zur API (wartet höchstens 30 s auf eine Antwort)
    api = httpx.Client(base_url=args.url, timeout=30)
    # Bricht ab, wenn API nicht läuft
    try:
        api.get("/health").raise_for_status()
    except httpx.HTTPError:
        raise SystemExit(f"API unter {args.url} nicht erreichbar - erst starten: "
                         ".venv/bin/python -m production.api")

    # Lädt neue Transaktionen und nimmt nur so viele wie gewünscht
    zeilen = pd.read_parquet(STREAM).iloc[:args.anzahl].to_dict("records")
    print(f"Sende {len(zeilen)} Transaktionen an {args.url} - Abbrechen mit Ctrl+C")

    # ==========================================================================
    # 3. SENDEN
    # ==========================================================================

    # Zähler für Zusammenfassung
    gesendet = fehler = alarme = gefunden = verdaechtig = 0
    start = time.perf_counter()
    try:
        for zeile in zeilen:
            t0 = time.perf_counter()
            # Schickt Transaktion an API; bei einem Fehler zählen und weitermachen
            try:
                antwort = api.post("/transaktion", json=anfrage(zeile))
                antwort.raise_for_status()
            except httpx.HTTPError:
                fehler += 1
                continue
            bewertung = antwort.json()
            gesendet += 1
            # Liest echtes Label (nur Simulator kennt es)
            label = int(zeile["Is_laundering"])
            verdaechtig += label

            # Alarm: sofort ausgeben, Analyst prüft Fall
            if bewertung["alarm"]:
                alarme += 1
                gefunden += label
                # Formatiert Betrag und Prozent im deutschen Format
                betrag = f"{zeile['Amount']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                prozent = f"{bewertung['wahrscheinlichkeit_prozent']:.1f}".replace(".", ",")
                print(f"⚠ ALARM {zeile['Timestamp']}  Konto {zeile['Sender_account']} → "
                      f"{zeile['Receiver_account']}  {betrag}  {zeile['Payment_type']}  {prozent} %  "
                      f"(laut Daten: {'Geldwäsche' if label else 'keine Geldwäsche'})")
                # Analyst reicht nach Prüfung echtes Label als Feedback nach (für Precision)
                api.post("/feedback", json={"transaktion_id": bewertung["transaktion_id"], "geldwaesche": bool(label)})

            # Zeigt alle 500 Transaktionen Fortschritt
            if gesendet % 500 == 0:
                print(f"  {gesendet} gesendet, {alarme} Alarme, {fehler} Fehler, "
                      f"{gesendet / (time.perf_counter() - start):.1f} Tx/s")
            # Wartet bei gesetzter Rate restliche Zeit bis zur nächsten Transaktion
            if args.rate:
                time.sleep(max(0.0, 1 / args.rate - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        # Ctrl+C beendet Senden, Zusammenfassung kommt trotzdem
        print("\nAbgebrochen.")

    # ==========================================================================
    # 4. ZUSAMMENFASSUNG – Alarme im Vergleich zu echten Labels
    # ==========================================================================

    dauer = time.perf_counter() - start
    print(f"\nGesendet: {gesendet}  Fehler: {fehler}  Dauer: {dauer:.0f} s  ({gesendet / dauer:.1f} Tx/s)")
    # Precision = Anteil der Alarme, die wirklich Geldwäsche waren
    print(f"Alarme: {alarme}  davon echte Geldwäsche: {gefunden}  "
          f"(Precision {gefunden / alarme:.1%})" if alarme else f"Alarme: 0")
    # Recall = Anteil der Geldwäschefälle, die erkannt wurden
    print(f"Geldwäschefälle gesendet: {verdaechtig}  davon erkannt: {gefunden}  "
          f"(Recall {gefunden / verdaechtig:.1%})" if verdaechtig else "Geldwäschefälle gesendet: 0")


if __name__ == "__main__":
    main()
