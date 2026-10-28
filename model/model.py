# ==============================================================================
# TRAINING UND VERGLEICH DER MODELLE FÜR DIE GELDWÄSCHEERKENNUNG
# ==============================================================================
# Vergleicht zwei Modelle auf denselben Daten:
#   Logistische Regression -> einfach und gut erklärbar (Basis)
#   LightGBM               -> erkennt auch nichtlineare Zusammenhänge (Hauptmodell)
# LightGBM-Modell wird für Betrieb in production/models gespeichert.
#
# Start (aus Projektordner):
#   .venv/bin/python -m model.model
# Optionaler Vergleichslauf mit künstlichen Fällen aus manuelle_anomalien.py
# (bewertet auf demselben echten Test, speichert KEIN Modell):
#   .venv/bin/python -m model.model --mit-anomalien

import sys
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Dateien, Aufteilung, Spaltenauswahl und LightGBM-Einstellungen stehen in konfiguration.py
from konfiguration import (ANOMALIEN_DATEI, ANTEIL_TRAIN, ANTEIL_VAL, EARLY_STOPPING, EINLAUF_TAGE, LIGHTGBM,
                           LR_WEGLASSEN, MERKMALE_DATEI, MODELLE, NICHT_MERKMAL)
# Länderrisiko – dieselbe Funktion wie in features.py
from model.features import laenderrisiko
from production.merkmale import meldung


# ==============================================================================
# PARAMETER
# ==============================================================================

# Vergleichslauf mit künstlichen Geldwäschefällen? (Schalter --mit-anomalien)
MIT_ANOMALIEN = "--mit-anomalien" in sys.argv


# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================

# Lädt alle Transaktionen mit Merkmalen (von features.py; Datentypen sind schon richtig)
meldung(f"1. Lese {ANOMALIEN_DATEI if MIT_ANOMALIEN else MERKMALE_DATEI} ...")
df = pd.read_parquet(ANOMALIEN_DATEI if MIT_ANOMALIEN else MERKMALE_DATEI)

# Vergleichslauf: legt künstliche Fälle beiseite, damit Split exakt gleich bleibt wie ohne
kuenstlich = df[df["Künstlich"] == 1].drop(columns="Künstlich") if MIT_ANOMALIEN else df.iloc[:0]
if MIT_ANOMALIEN:
    df = df[df["Künstlich"] == 0].drop(columns="Künstlich").reset_index(drop=True)

meldung(f"Geladen: {len(df):,} Zeilen".replace(",", "."))


# ==============================================================================
# 2. EINLAUFPHASE UND ZEITLICHER SPLIT
# Im ersten Monat fehlt jede Vorgeschichte – fast jede Verbindung wirkt "neu".
# Zeitlich statt zufällig splitten: Modell lernt aus Vergangenheit und
# wird an Zukunft geprüft, so wie im echten Betrieb.
# ==============================================================================

meldung("2. Einlaufphase entfernen und zeitlich splitten ...")
# Entfernt ersten Monat
start = df["Timestamp"].min() + pd.Timedelta(days=EINLAUF_TAGE)
df = df[df["Timestamp"] >= start].reset_index(drop=True)

# Berechnet Schnittstellen (Zeilen sind chronologisch sortiert)
grenze_train = int(len(df) * ANTEIL_TRAIN)
grenze_val = int(len(df) * (ANTEIL_TRAIN + ANTEIL_VAL))

# Schneidet in drei aufeinanderfolgende Zeiträume
train = df.iloc[:grenze_train].copy()
val = df.iloc[grenze_train:grenze_val].copy()
test = df.iloc[grenze_val:].copy()

# Vergleichslauf: künstliche Fälle kommen NUR ins Training – Validierung und Test bleiben echt
if MIT_ANOMALIEN:
    train = pd.concat([train, kuenstlich], ignore_index=True)
    meldung(f"Vergleichslauf: {len(kuenstlich):,} künstliche Transaktionen im Training".replace(",", "."))

# Zeigt Zeitraum, Größe und Geldwäschefälle je Teil
for name, teil in [("Train", train), ("Validierung", val), ("Test", test)]:
    print(f"  {name:12s} {teil['Timestamp'].min():%d.%m.%Y} bis {teil['Timestamp'].max():%d.%m.%Y}  "
          f"{len(teil):>9,} Zeilen  {int(teil['Is_laundering'].sum()):>5,} Geldwäschefälle".replace(",", "."))


# ==============================================================================
# 3. LÄNDERRISIKO NUR AUS DEN TRAININGSDATEN
# Länderrisiko nutzt Label. features.py hat es aus ALLEN Daten berechnet –
# hier wird es nur aus Train neu berechnet, sonst nutzt Modell Wissen aus
# Testzeitraum und wirkt zu gut.
# ==============================================================================

meldung("3. Länderrisiko aus den Trainingsdaten ...")
# Anteil Geldwäsche im Training (für unbekannte Länder)
gesamt_quote = train["Is_laundering"].mean()
# Geglättete Quote je Empfänger- und Senderland, nur aus Train
quote_empfaenger = laenderrisiko(train, "Receiver_bank_location")
quote_sender = laenderrisiko(train, "Sender_bank_location")

# Überträgt die Quoten auf alle drei Teile
for teil in (train, val, test):
    teil["Risiko_Land_Empfänger"] = teil["Receiver_bank_location"].map(quote_empfaenger).fillna(gesamt_quote)
    teil["Risiko_Land_Sender"] = teil["Sender_bank_location"].map(quote_sender).fillna(gesamt_quote)

# Trennt Merkmale (X) und Label (y) für alle drei Teile
X_train, y_train = train.drop(columns=NICHT_MERKMAL), train["Is_laundering"]
X_val, y_val = val.drop(columns=NICHT_MERKMAL), val["Is_laundering"]
X_test, y_test = test.drop(columns=NICHT_MERKMAL), test["Is_laundering"]


# ==============================================================================
# 4. LOGISTISCHE REGRESSION
# Lineares Modell: Jedes Merkmal erhöht oder senkt Risiko um einen festen
# Faktor. Gut erklärbar, braucht aber skalierte Merkmale und Klassengewichte.
# ==============================================================================

meldung(f"4. Logistische Regression trainieren ({len(X_train):,} Zeilen) ...".replace(",", "."))
# Alle Merkmale außer oben genannten Ausnahmen
spalten_lr = [s for s in X_train.columns if s not in LR_WEGLASSEN]

# Bringt alle Merkmale auf eine vergleichbare Skala (gelernt NUR auf Train)
skalierer = StandardScaler().fit(X_train[spalten_lr])
# "balanced": Ohne Gewichtung würde Modell bei 0,1 % Geldwäsche fast immer "nein" sagen
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(skalierer.transform(X_train[spalten_lr]), y_train)

# Wahrscheinlichkeiten für Validierung und Test
p_val_lr = lr.predict_proba(skalierer.transform(X_val[spalten_lr]))[:, 1]
p_test_lr = lr.predict_proba(skalierer.transform(X_test[spalten_lr]))[:, 1]
meldung("Logistische Regression trainiert")


# ==============================================================================
# 5. LIGHTGBM
# Viele kleine Entscheidungsbäume, die nacheinander Fehler der vorherigen
# korrigieren (Gradient Boosting). Erkennt Kombinationen selbst, keine Skalierung nötig.
# ==============================================================================

meldung("5. LightGBM trainieren (Zwischenstand alle 100 Bäume) ...")
gbm = lgb.LGBMClassifier(
    **LIGHTGBM,                 # Lernrate, Blätter usw. (siehe PARAMETER)
    subsample_freq=1,           # nötig, damit "subsample" wirkt
    metric="average_precision", # PR-AUC als Stoppkriterium (eingebaut = doppelt so schnell)
    random_state=42,            # gleiches Ergebnis bei jedem Lauf
    verbosity=-1,               # keine internen Meldungen
)
# Trainiert und stoppt, sobald sich PR-AUC auf Validierung nicht mehr verbessert
gbm.fit(
    X_train, y_train,
    eval_X=X_val, eval_y=y_val,
    callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
               lgb.log_evaluation(100)],   # zeigt alle 100 Bäume PR-AUC
)

# Wahrscheinlichkeiten für Validierung und Test
p_val_gbm = gbm.predict_proba(X_val)[:, 1]
p_test_gbm = gbm.predict_proba(X_test)[:, 1]
meldung(f"LightGBM trainiert ({gbm.best_iteration_} Bäume)")


# ==============================================================================
# 6. SCHWELLE WÄHLEN UND BEWERTEN
# Ab welcher Wahrscheinlichkeit ein Alarm kommt, wird auf Validierung
# entschieden (bestes F1) und erst danach einmal auf Test geprüft.
#
# Kennzahlen zur Erkennung (entscheidend bei 0,1 % Geldwäsche):
#   PR-AUC    -> Gesamtgüte über alle Schwellen; Zufall = Geldwäschequote
#   Recall    -> Anteil der gefundenen Geldwäschefälle
#   Precision -> Anteil echter Fälle unter Alarmen
# Allgemeine Werte (zum Vergleich mit üblichen Angaben):
#   Accuracy          -> irreführend: "immer nein" erreicht schon ~99,9 %
#   Balanced Accuracy -> gewichtet beide Klassen gleich; "immer nein" = 0,5
#   F1                -> Ausgleich zwischen Precision und Recall
#   ROC-AUC           -> wirkt bei seltener Geldwäsche oft zu optimistisch
# ==============================================================================

# Findet Schwelle mit bestem F1-Wert
def beste_schwelle(y, p):
    precision, recall, schwellen = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return schwellen[np.argmax(f1[:-1])]


# Schreibt eine ganze Zahl mit Tausenderpunkt, z. B. 1.532
def zahl(n):
    return f"{n:,}".replace(",", ".")


# Schreibt eine Quote lesbar als "X von Y (Z %)"
def anteil(teil, gesamt):
    prozent = f"{100 * teil / gesamt:.1f}".replace(".", ",") + " %" if gesamt else "–"
    return f"{zahl(teil)} von {zahl(gesamt)} ({prozent})"


# Berechnet alle Kennzahlen eines Modells auf Test
def bewerten(name, y, p, schwelle):
    # 1 = Alarm, wenn Wahrscheinlichkeit Schwelle erreicht
    alarm = (p >= schwelle).astype(int)
    # Zählt richtig/falsch erkannte Fälle (tn = richtig nein, fp = Fehlalarm, fn = verpasst, tp = gefunden)
    tn, fp, fn, tp = confusion_matrix(y, alarm, labels=[0, 1]).ravel()
    return {
        "Modell": name,
        # Erkennung
        "PR-AUC": round(average_precision_score(y, p), 4),
        "Schwelle": round(float(schwelle), 4),
        "Recall": round(tp / (tp + fn), 3),
        "Precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
        "gefunden": int(tp),
        "verpasst": int(fn),
        "Fehlalarme": int(fp),
        # Allgemeine Genauigkeitswerte
        "Accuracy": round(accuracy_score(y, alarm), 5),
        "Balanced Accuracy": round(balanced_accuracy_score(y, alarm), 3),
        "F1": round(f1_score(y, alarm, zero_division=0), 3),
        "ROC-AUC": round(roc_auc_score(y, p), 3),
        # Übersicht in Zahlen
        "Fälle gefunden": anteil(int(tp), int(tp + fn)),
        "Alarme richtig": anteil(int(tp), int(tp + fp)),
    }


meldung("6. Schwelle wählen und auf dem Test bewerten ...")
# Wählt Schwellen auf Validierung
schwelle_lr = beste_schwelle(y_val, p_val_lr)
schwelle_gbm = beste_schwelle(y_val, p_val_gbm)

# Bewertet alle Modelle auf Test
ergebnis = pd.DataFrame([
    # Vergleichszeile: sagt immer "keine Geldwäsche"
    bewerten("Immer 'keine Geldwäsche'", y_test, np.zeros(len(y_test)), 0.5),
    bewerten("Logistische Regression", y_test, p_test_lr, schwelle_lr),
    bewerten("LightGBM", y_test, p_test_gbm, schwelle_gbm),
])

# Spaltengruppen für Ausgabe
ERKENNUNG = ["Modell", "PR-AUC", "Schwelle", "Recall", "Precision", "gefunden", "verpasst", "Fehlalarme"]
GENAUIGKEIT = ["Modell", "Accuracy", "Balanced Accuracy", "F1", "ROC-AUC"]

# Gibt Ergebnisse als Tabellen aus
print(f"\nERGEBNIS AUF DEM TEST (Zufall erreicht PR-AUC {y_test.mean():.4f})")
print("\nErkennung von Geldwäsche:")
print(ergebnis[ERKENNUNG].to_string(index=False))
print("\nAllgemeine Genauigkeitswerte:")
print(ergebnis[GENAUIGKEIT].to_string(index=False))
print("\nÜbersicht in Zahlen:")
print(ergebnis[["Modell", "Fälle gefunden", "Alarme richtig"]].to_string(index=False))

# Zeigt, welche Merkmale LightGBM am meisten nutzt (Anteil am Gesamtgewinn)
wichtigkeit = pd.Series(gbm.booster_.feature_importance("gain"), index=X_train.columns)
print("\nMerkmalswichtigkeit LightGBM (Anteil am Gesamtgewinn):")
print((100 * wichtigkeit / wichtigkeit.sum()).sort_values(ascending=False).round(1).to_string())


# Vergleichslauf endet hier: Betriebsmodell bleibt unverändert
if MIT_ANOMALIEN:
    meldung("Vergleichslauf mit künstlichen Fällen – Modell wird NICHT gespeichert.")
    sys.exit()


# ==============================================================================
# 7. MODELLARTEFAKT SPEICHERN
# Training ist teuer und selten, Vorhersagen passieren bei jeder Transaktion.
# Deshalb wird alles gespeichert, was im Betrieb genauso gelten muss:
#   merkmale     -> Spalten in genau der Reihenfolge, die Modell erwartet
#   schwelle     -> ab dieser Wahrscheinlichkeit gibt es einen Alarm
#   Länderrisiko -> im Betrieb nicht neu berechenbar (kein Label)
#   version      -> nachvollziehbar, welches Modell einen Alarm auslöste
# Gespeichert wird nur LightGBM; logistische Regression dient zum Vergleich.
# Jedes Training erzeugt eine NEUE Datei, Name beginnt mit Zeitstempel –
# alphabetisch letzte Datei ist also immer neueste.
# ==============================================================================

meldung("7. Modellartefakt speichern ...")
# Dateiname mit Zeitstempel, z. B. lightgbm_2026-10-18_12-30-00.joblib
zeitpunkt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
modell_datei = MODELLE / f"lightgbm_{zeitpunkt}.joblib"

# Packt Modell und alles Zugehörige zusammen
artefakt = {
    "modell": gbm,
    "merkmale": list(X_train.columns),
    "schwelle": float(schwelle_gbm),
    "risiko_land_empfaenger": quote_empfaenger.to_dict(),
    "risiko_land_sender": quote_sender.to_dict(),
    "gesamt_quote": float(gesamt_quote),
    "version": f"lgbm-{zeitpunkt}",
}

# Speichert alles als eine Datei in production/models
MODELLE.mkdir(exist_ok=True)
joblib.dump(artefakt, modell_datei)
meldung(f"Modell gespeichert: {modell_datei.name}")


# ==============================================================================
# NOTIZ: SO LÄDT DIE API DAS MODELL (production/api.py)
# ==============================================================================
#   neuestes = max(MODELLE.glob("lightgbm_20*.joblib"))   # neueste Datei
#   artefakt = joblib.load(neuestes)
#   wahrscheinlichkeit = artefakt["modell"].predict_proba(X_neu[artefakt["merkmale"]])[:, 1]
#   alarm = wahrscheinlichkeit >= artefakt["schwelle"]
