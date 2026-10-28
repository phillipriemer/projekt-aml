# ==============================================================================
# MERKMALSBERECHNUNG – EIN Code für Training UND Betrieb
# ==============================================================================
# Genutzt von production/datenbank.py (jede neue Transaktion) und
# model/features.py (ganzer Datensatz). Weil beide diese Datei nutzen, rechnen
# Training und API garantiert gleich (geprüft im Paritätstest).
#
# Grundidee: Geldwäsche fällt selten an EINER Transaktion auf, sondern am
# Verhalten eines Kontos über längere Zeit. Es wird nur Vergangenheit genutzt,
# nie Zukunft (im echten Betrieb ebenfalls unbekannt).
#
# Rohspalten heißen wie im Kaggle-Datensatz SAML-D (Sender_account, Amount, ...),
# selbst berechnete Merkmale deutsch (Z_Sender, Fan_Out_24h, ...).

import time

import numpy as np
import pandas as pd

# Bargeld-Schwelle, längstes Zeitfenster und Zahlungsarten stehen in konfiguration.py
from konfiguration import LAENGSTES_FENSTER, T, ZAHLUNGSARTEN

# Merkt sich Programmstart, damit Meldungen Laufzeit anzeigen können
START = time.perf_counter()


# Gibt eine Zwischenmeldung mit Laufzeit aus (flush = sofort sichtbar)
def meldung(text, melden=True):
    if melden:
        print(f"[{time.perf_counter() - START:7.1f} s] {text}", flush=True)


# ==============================================================================
# MERKMALE BERECHNEN
# ==============================================================================

# Berechnet alle Merkmale außer Länderrisiko und Zahlungsart-Spalten.
# Erwartet Rohspalten (siehe ROHSPALTEN in datenbank.py), Zeilen chronologisch nach Index geordnet.
# melden=False schaltet Meldungen ab (API ruft das pro Transaktion auf).
def merkmale_berechnen(df, T=T, melden=False):
    # Arbeitet auf einer Kopie, damit Originaldaten unverändert bleiben
    df = df.copy()


    # --------------------------------------------------------------------------
    # 1. ZEITPUNKT
    # Nachts und am Wochenende wird weniger kontrolliert
    # --------------------------------------------------------------------------

    meldung("1. Zeitpunkt: Wochenende, Stunde, Nacht ...", melden)

    # 1 = Samstag oder Sonntag
    df["Wochenende"] = (df["Timestamp"].dt.day_of_week >= 5).astype(int)
    # Uhrzeit als Zahl von 0 bis 23
    df["Stunde"] = df["Timestamp"].dt.hour
    # 1 = zwischen 23 Uhr und 5 Uhr
    df["Nacht"] = ((df["Stunde"] >= 23) | (df["Stunde"] < 5)).astype(int)


    # --------------------------------------------------------------------------
    # 2. BETRAG
    # Entscheidend ist weniger Höhe als Abweichung vom Gewohnten
    # --------------------------------------------------------------------------

    meldung("2. Betrag: Logarithmus ...", melden)

    # Staucht wenige riesige Beträge auf eine vergleichbare Skala
    df["Betrag_Log"] = np.log(df["Amount"])

    # Z-Score = Wie ungewöhnlich ist Betrag für DIESES Konto (letzte 30 Tage)?
    #   closed="left" -> nur frühere Transaktionen, nicht aktuelle
    #   keine Vorgeschichte oder Streuung 0 -> 0, Extremwerte auf ±20 begrenzt

    meldung("2. Betrag: Z-Score Sender (30 Tage) ...", melden)
    # Sortiert nach Sender und Zeit, damit Zeitfenster je Konto läuft
    df = df.sort_values(["Sender_account", "Timestamp"], kind="stable")
    # Gleitendes 30-Tage-Fenster über frühere Beträge jedes Senders
    fenster = df.groupby("Sender_account").rolling(LAENGSTES_FENSTER, on="Timestamp", closed="left")["Amount"]
    # Abstand zum Durchschnitt, gemessen in Standardabweichungen
    z = (df["Amount"] - fenster.mean().to_numpy()) / fenster.std().to_numpy()
    # Ersetzt ungültige Werte durch 0 und kappt Ausreißer
    df["Z_Sender"] = z.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-20, 20)

    meldung("2. Betrag: Z-Score Empfänger (30 Tage) ...", melden)
    # Dasselbe für Eingänge jedes Empfängers
    df = df.sort_values(["Receiver_account", "Timestamp"], kind="stable")
    fenster = df.groupby("Receiver_account").rolling(LAENGSTES_FENSTER, on="Timestamp", closed="left")["Amount"]
    z = (df["Amount"] - fenster.mean().to_numpy()) / fenster.std().to_numpy()
    df["Z_Empfänger"] = z.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-20, 20)


    # --------------------------------------------------------------------------
    # 3. BARGELD UND STRUCTURING
    # Täter stückeln Bargeld, damit jeder Betrag knapp unter Schwelle bleibt
    # --------------------------------------------------------------------------

    meldung("3. Bargeld: Structuring (7 Tage) ...", melden)

    # Markiert alle Bargeld-Transaktionen (Ein- und Auszahlung)
    bar = df["Payment_type"].isin(["Cash Deposit", "Cash Withdrawal"])

    # Structuring: Einzelbetrag unter T, aber Bargeld-Summe (letzte 7 Tage) über T
    # (offen: bei Einzahlungen wäre evtl. Empfänger richtiges Konto)
    # Nur Bargeld, sortiert nach Sender und Zeit
    b = df[bar].sort_values(["Sender_account", "Timestamp"])
    # Bargeld-Summe (letzte 7 Tage) je Sender (inklusive aktueller Transaktion)
    summe = b.groupby("Sender_account").rolling("7D", on="Timestamp")["Amount"].sum().to_numpy()
    # 1 = Einzelbetrag unter T, Summe über T
    df["Bar_Structuring_7d"] = df.index.isin(b.index[(b["Amount"] < T) & (summe > T)]).astype(int)


    # --------------------------------------------------------------------------
    # 4. GELDFLUSS: BAREINZAHLUNG KURZ DAVOR
    # Bargeld einzahlen und schnell weiterleiten (oft ins Ausland) verwischt Spur.
    # Ob es ins Ausland geht, lernt Modell über Länderrisiko und Währung.
    # Weiterleitendes Konto = EMPFÄNGER der Bareinzahlung.
    # --------------------------------------------------------------------------

    meldung("4. Geldfluss: Bareinzahlung in den 24 h davor ...", melden)

    # Sammelt alle Bareinzahlungen mit Zeitpunkt und begünstigtem Konto
    einzahlung = df.loc[df["Payment_type"] == "Cash Deposit", ["Timestamp", "Receiver_account"]]
    # Benennt so um, dass begünstigtes Konto später als Sender gefunden wird
    einzahlung = einzahlung.rename(columns={"Receiver_account": "Sender_account", "Timestamp": "Einzahlung_Zeit"})
    # merge_asof braucht zeitlich sortierte Daten
    einzahlung = einzahlung.sort_values("Einzahlung_Zeit")

    # Merkt sich Zeilennummer, damit Ergebnis zurückgeordnet werden kann
    links = df[["Timestamp", "Sender_account"]].rename_axis("Zeile").reset_index().sort_values("Timestamp")
    # Sucht je Konto letzte Bareinzahlung innerhalb 24 h davor
    m = pd.merge_asof(links, einzahlung, left_on="Timestamp", right_on="Einzahlung_Zeit",
                      by="Sender_account", tolerance=pd.Timedelta("24h"), allow_exact_matches=False)
    # 1 = es gab eine solche Einzahlung
    df["Bar_davor_24h"] = pd.Series(m["Einzahlung_Zeit"].notna().astype(int).to_numpy(), index=m["Zeile"])

    # Stellt chronologische Reihenfolge wieder her (für Abschnitt 5 nötig)
    df = df.sort_index()


    # --------------------------------------------------------------------------
    # 5. KONTOBEZIEHUNGEN
    # Neue Verbindungen (Strohleute) und viele Gegenkonten sind verdächtig:
    # Geld wird verteilt (Fan-out) oder gesammelt (Fan-in)
    # --------------------------------------------------------------------------

    meldung("5. Kontobeziehungen: neue Sender-Empfänger-Paare ...", melden)

    # 1 = erste Überweisung zwischen genau diesem Sender und Empfänger.
    # API sieht nur 30 Tage und korrigiert das mit allen bekannten Paaren.
    df["Neuer_Sender_Empfänger"] = (~df.duplicated(["Sender_account", "Receiver_account"])).astype(int)

    # Fan-out = an wie viele verschiedene Konten hat Sender zuletzt gezahlt?
    # Fan-in  = von wie vielen verschiedenen Konten hat Empfänger Geld bekommen?
    # In SAML-D sind hohe Werte meist normale Konten (z. B. Gehälter) – LightGBM
    # kann diesen ungleichmäßigen Zusammenhang abbilden. Zwei Fenster, weil
    # langsames Verteilen erst über mehrere Tage sichtbar wird.
    meldung("5. Kontobeziehungen: Fan-out 24 h ...", melden)
    df["Fan_Out_24h"] = anzahl_gegenkonten(df, "Sender_account", "Receiver_account", "24h")
    meldung("5. Kontobeziehungen: Fan-out 7 Tage ...", melden)
    df["Fan_Out_7d"] = anzahl_gegenkonten(df, "Sender_account", "Receiver_account", "7D")
    meldung("5. Kontobeziehungen: Fan-in 24 h ...", melden)
    df["Fan_In_24h"] = anzahl_gegenkonten(df, "Receiver_account", "Sender_account", "24h")
    meldung("5. Kontobeziehungen: Fan-in 7 Tage ...", melden)
    df["Fan_In_7d"] = anzahl_gegenkonten(df, "Receiver_account", "Sender_account", "7D")


    # --------------------------------------------------------------------------
    # 6. WÄHRUNG
    # Ein Währungswechsel kann Spur zusätzlich verschleiern
    # --------------------------------------------------------------------------

    meldung("6. Währung: Abweichung ...", melden)

    # 1 = Zahlungs- und Empfangswährung sind verschieden
    df["Waehrung_Abweichung"] = (df["Payment_currency"] != df["Received_currency"]).astype(int)

    return df


# ==============================================================================
# HILFSFUNKTION: VERSCHIEDENE GEGENKONTEN ZÄHLEN (für Fan-in / Fan-out)
# ==============================================================================

# Zählt je Transaktion, mit wie vielen VERSCHIEDENEN Gegenkonten ein Konto im
# Zeitfenster davor zu tun hatte (aktuelle und gleichzeitige zählen nicht mit).
# Idee: Jede Zahlung zählt ab ihrem Zeitpunkt mit, bis Fenster abläuft oder
# dasselbe Gegenkonto erneut bezahlt wird. Anzahl zum Zeitpunkt t =
# begonnene Zählzeiten vor t minus beendete Zählzeiten vor t.
def anzahl_gegenkonten(df, konto, gegenkonto, fenster):
    # Rechnet Zeitstempel in Sekunden seit Datensatzbeginn um
    sekunden = (df["Timestamp"] - df["Timestamp"].min()).dt.total_seconds().to_numpy(dtype="int64")
    # Fensterbreite in Sekunden (z. B. 24 h = 86.400)
    breite = int(pd.Timedelta(fenster).total_seconds())

    # Sortiert nach Konto, Gegenkonto und Zeit
    reihenfolge = np.lexsort((sekunden, df[gegenkonto].to_numpy(), df[konto].to_numpy()))
    k = df[konto].to_numpy()[reihenfolge]
    g = df[gegenkonto].to_numpy()[reihenfolge]
    s = sekunden[reihenfolge]

    # Findet Zeilen, auf die eine Zahlung an dasselbe Gegenkonto folgt
    gleiches_paar = (k[1:] == k[:-1]) & (g[1:] == g[:-1])
    # Zeitpunkt nächster Zahlung an dasselbe Gegenkonto ("nie" = größte Zahl)
    naechste = np.full(len(s), np.iinfo(np.int64).max)
    naechste[:-1][gleiches_paar] = s[1:][gleiches_paar]
    # Zählzeit endet bei nächster Zahlung oder am Fensterende
    ende = np.minimum(naechste, s + breite)

    # Gibt jedem Konto einen eigenen Zahlenbereich, damit sich Konten nicht vermischen
    nummer = pd.factorize(k)[0].astype("int64")
    bereich = int(sekunden.max()) + breite + 1
    beginn_sortiert = np.sort(nummer * bereich + s)
    ende_sortiert = np.sort(nummer * bereich + ende)
    zeitpunkt = nummer * bereich + s

    # Begonnene minus beendete Zählzeiten = Anzahl verschiedener Gegenkonten
    anzahl = (np.searchsorted(beginn_sortiert, zeitpunkt, "left")
              - np.searchsorted(ende_sortiert, zeitpunkt, "left"))

    # Ordnet Ergebnis zurück in ursprüngliche Zeilenreihenfolge
    ergebnis = np.empty(len(df), dtype="int64")
    ergebnis[reihenfolge] = anzahl
    return ergebnis


# ==============================================================================
# 7. ZAHLUNGSART
# Eine 0/1-Spalte je Zahlungsart (One-Hot), weil Arten keine Reihenfolge haben
# ==============================================================================

# Hängt je Zahlungsart eine Spalte "Zahlung_Typ_<Art>" an; Textspalte bleibt
def zahlungsart_spalten(df):
    df = df.copy()
    for art in ZAHLUNGSARTEN:
        # 1 = Transaktion hat genau diese Zahlungsart
        df[f"Zahlung_Typ_{art}"] = (df["Payment_type"] == art).astype("int8")
    return df
