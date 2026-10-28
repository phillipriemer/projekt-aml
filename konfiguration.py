# ==============================================================================
# KONFIGURATION – alle einstellbaren Werte des Projekts an EINER Stelle
# ==============================================================================
# Training (model/) und Betrieb (production/) lesen ihre Werte von hier, z. B.:
#   from konfiguration import T, STREAM_ANZAHL
# Neue Vorgabe (z. B. andere Bargeld-Schwelle) oder neue Datenquelle (andere
# Zahlungsarten, anderer Zeitraum) -> Wert hier ändern, Pipeline neu laufen lassen.
# Liegt auch im Docker-Image, weil API und Merkmalsberechnung sie brauchen.

from pathlib import Path


# ==============================================================================
# 1. DATEIEN
# ==============================================================================

# Projektordner (dieser Ordner; im Container: /app)
PROJEKT = Path(__file__).resolve().parent

# Die letzten so vielen Transaktionen schickt Simulator als "neue" Transaktionen
# (Name der Stream-Datei passt sich automatisch an: 100.000 -> "0.1mio")
STREAM_ANZAHL = 100_000

# Training (nicht im Container)
ROHDATEN = PROJEKT / "model" / "data" / "SAML-D.csv"                          # Kaggle-Datensatz
MERKMALE_DATEI = PROJEKT / "model" / "data" / "SAML-D_mod.parquet"            # alle Transaktionen mit Merkmalen
ANOMALIEN_DATEI = PROJEKT / "model" / "data" / "SAML-D_mod_mit_anomalien.parquet"  # optional: plus künstliche Fälle

# Betrieb
# Vorgeschichte der API: alle außer letzten STREAM_ANZAHL (9,4 Mio. – bei anderer STREAM_ANZAHL Namen anpassen)
BASIS = PROJEKT / "production" / "data" / "SAML-D_9.4mio.parquet"
# Neue Transaktionen für Simulator: letzte STREAM_ANZAHL
STREAM = PROJEKT / "production" / "data" / f"SAML-D_last_{STREAM_ANZAHL / 1e6:g}mio.parquet"
DATENBANK = PROJEKT / "production" / "data" / "aml.db"                                   # Datenbank der API
MODELLE = PROJEKT / "production" / "models"                                              # model.py speichert, API lädt neuestes


# ==============================================================================
# 2. DATEN UND MERKMALE (gelten für Training UND Betrieb)
# ==============================================================================

# Alle Zahlungsarten im Datensatz – API nimmt nur diese an, je Art entsteht eine 0/1-Spalte
ZAHLUNGSARTEN = ["ACH", "Cash Deposit", "Cash Withdrawal", "Cheque", "Credit card", "Cross-border", "Debit card"]

# Schwelle für Bargeld-Prüfungen (Structuring: Einzelbeträge darunter, Summe darüber)
T = 10000

# Längstes Zeitfenster der Merkmale – danach richtet sich, wie viel Vorgeschichte API je Transaktion holt
LAENGSTES_FENSTER = "30D"


# ==============================================================================
# 3. TRAINING
# ==============================================================================

# Erster Monat wird nicht verwendet (noch keine Vorgeschichte)
EINLAUF_TAGE = 30
# 70 % Training, 15 % Validierung, Rest (15 %) Test – zeitlich, nicht zufällig
ANTEIL_TRAIN = 0.70
ANTEIL_VAL = 0.15

# Glättung beim Länderrisiko: so stark zählt Gesamtquote mit
K = 1000

# Diese Spalten gehen NICHT ins Modell:
#   Is_laundering            -> Zielvariable
#   Timestamp, Länder        -> nur für Split und Länderrisiko gebraucht
#   Konto-IDs                -> Modell würde Konten auswendig lernen statt Muster
#   Währungen (Text)         -> ersetzt durch Waehrung_Abweichung
#   Amount                   -> ersetzt durch Betrag_Log
#   Zahlung_Typ_Cross-border -> zu stark korreliert mit Waehrung_Abweichung (r = 0,91)
NICHT_MERKMAL = ["Is_laundering", "Timestamp", "Sender_bank_location", "Receiver_bank_location",
                 "Sender_account", "Receiver_account", "Payment_currency", "Received_currency",
                 "Amount", "Zahlung_Typ_Cross-border"]

# Spalten, die nur logistische Regression weglässt:
#   Zahlung_Typ_ACH -> eine Zahlungsart muss als Vergleichsgruppe wegfallen (Dummy-Falle)
#   Stunde          -> 23 Uhr und 0 Uhr liegen nebeneinander, als Zahl aber weit auseinander
LR_WEGLASSEN = ["Zahlung_Typ_ACH", "Stunde"]

# Training stoppt, wenn sich PR-AUC so viele Runden nicht verbessert
EARLY_STOPPING = 50

# Beste Einstellung aus Random Search (20 Versuche, nur auf Validierung bewertet):
# PR-AUC 0,747 statt 0,662. Keine Klassengewichtung – beste Versuche hatten alle keine.
LIGHTGBM = dict(
    n_estimators=3000,       # Obergrenze, Early Stopping bricht vorher ab (ca. 1.500 Bäume)
    learning_rate=0.01,      # kleine Schritte, dafür viele Bäume
    num_leaves=63,           # mittlere Komplexität je Baum
    min_child_samples=100,   # mind. 100 Zeilen je Blatt, kein Auswendiglernen
    colsample_bytree=0.9,    # jeder Baum sieht 90 % der Merkmale
    subsample=0.9,           # jeder Baum sieht 90 % der Zeilen
    reg_lambda=5,            # bremst zu große Werte in Blättern
)


# ==============================================================================
# 4. BETRIEB (API, Monitoring)
# ==============================================================================

# Adresse der API (lokal; im Container startet sie auf 0.0.0.0, siehe Dockerfile)
HOST, PORT = "127.0.0.1", 8000
# Seite /alarme lädt sich alle so viele Sekunden neu
NEU_LADEN_S = 2
# Drift: unter 0,10 "stabil", ab 0,25 "Drift", dazwischen "beobachten"
PSI_STABIL, PSI_DRIFT = 0.10, 0.25
# Drift-Referenz: Beträge der so vielen Tage vor den neuen Transaktionen
REFERENZ_TAGE = 30
