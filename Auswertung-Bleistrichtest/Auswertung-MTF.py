"""
Autor 1: Hakim Kayed

Verwendungszweck: Masterarbeit zur Untersuchung der Bildqualität von Angiografiebildern
Hinweis / Zitat: Erstellt, übersetzt und optimiert unter Nutzung von Google Gemini 3.1 Pro 
                 zur Übersetzung von Konzepten/Ideen in Python-Code sowie zur Code-Optimierung.
Zweck: Automatische Auswertung von MTF-Kurven, Kennfrequenzen (MTF50, MTF10) und Integralen (AUC) 
       mittels PyImageJ, Signalverarbeitung und Pchip-Spline-Interpolation.
"""

import os
import sys
import glob
import shlex
import shutil
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import imagej
import scyjava as sj
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator

# Setup für Java-Umgebung
sj.config.set_java_constraints(version="21", vendor="temurin")


def load_macro_code():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    macro_path = os.path.join(script_dir, "Macro-plot.ijm")

    if not os.path.exists(macro_path):
        raise FileNotFoundError(f"Makro-Datei nicht gefunden unter: {macro_path}")

    with open(macro_path, "r", encoding="utf-8") as f:
        return f.read()


def komma(x_val, pos):
    return str(x_val).replace(".", ",")


# HINWEIS ZUR REIHENFOLGE DER FUNKTIONEN:
# Python führt Skripte von oben nach unten aus. Daher müssen die mathematischen und 
# bildverarbeitenden Funktionen (Schritt 3 & Schritt 2) hier oben definiert werden, 
# damit das Hauptprogramm ab Schritt 1 weiter unten auf sie zugreifen kann.
# Der tatsächliche Programmablauf startet weiter unten bei SCHRITT 1.


# SCHRITT 3: SIGNALVERARBEITUNG & MTF-BERECHNUNG
def calculate_mtf(df_combined, image_name="Bild", debug_dir=None):
    x_spalte = "X"
    x = df_combined[x_spalte].to_numpy()
    y = df_combined["Grauwert"].to_numpy()

    # Untergrundkorrektur (Rolling Quantile & Polynomial Fit)
    window_size = 25
    s_y = pd.Series(y)
    untergrund_10prozent = s_y.rolling(window=window_size, center=True, min_periods=1).quantile(0.10)

    poly_grad = 3
    koeffizienten = np.polyfit(x, untergrund_10prozent, poly_grad)
    untergrund_funktion = np.poly1d(koeffizienten)

    y_untergrund = untergrund_funktion(x)
    y_korrigiert = y - y_untergrund

    # Dynamische Schwelle & Peak-Finder
    y_max_global = np.max(y_korrigiert)
    schwelle_start = 0.50 * y_max_global
    schwelle_ende = 0.15 * y_max_global
    dynamic_threshold = np.linspace(schwelle_start, schwelle_ende, len(y_korrigiert))

    raw_peaks, _ = find_peaks(y_korrigiert, height=0)
    valid_peaks = np.array([p for p in raw_peaks if y_korrigiert[p] >= dynamic_threshold[p]])

    # Nahe Peaks adaptiv zusammenführen
    ERWARTETE_GRUPPEN = 20
    schrittweite = x[1] - x[0] if len(x) > 1 else 1.0
    x_min, x_max = x[0], x[-1]
    x_gesamtlänge = x_max - x_min if x_max > x_min else 1.0

    geschätzte_gruppenbreite_x = x_gesamtlänge / ERWARTETE_GRUPPEN
    
    merged_peaks = []
    if len(valid_peaks) > 0:
        current_cluster = [valid_peaks[0]]
        
        for p in valid_peaks[1:]:
            p_prev = current_cluster[-1]
            abstand_x = x[p] - x[p_prev]
            
            limit_x = max(schrittweite * 2.5, geschätzte_gruppenbreite_x * 0.06)
            
            if abstand_x <= limit_x:
                current_cluster.append(p)
            else:
                best_peak = current_cluster[np.argmax(y_korrigiert[current_cluster])]
                merged_peaks.append(best_peak)
                current_cluster = [p]
        
        best_peak = current_cluster[np.argmax(y_korrigiert[current_cluster])]
        merged_peaks.append(best_peak)

    peaks_max_idx = np.array(merged_peaks)

    # Helferfunktion: Echtes Tal finden
    def finde_echtes_minimum(p_start, p_end, y_data):
        segment = y_data[p_start : p_end + 1]
        if len(segment) <= 2:
            return p_start + np.argmin(segment)
            
        rand_offset = max(1, int(len(segment) * 0.15))
        kern_segment = segment[rand_offset : len(segment) - rand_offset]
        
        if len(kern_segment) > 0:
            min_rel_idx = rand_offset + np.argmin(kern_segment)
        else:
            min_rel_idx = np.argmin(segment)
            
        return p_start + min_rel_idx

    # Dynamische Peak- & Tal-Erkennung für 20 Linienpaargruppen
    gruppen = []
    vorherige_breite_x = None
    gruppe_abgebrochen = False

    def ist_im_erlaubten_bereich(g_idx, x_pos):
        rel_pos = (x_pos - x_min) / x_gesamtlänge
        if 0 <= g_idx <= 7:        # Gruppe 1 bis 8 (0 bis 50%)
            return rel_pos <= 0.50
        elif 8 <= g_idx <= 11:     # Gruppe 9 bis 12 (50 bis 67%)
            return 0.50 <= rel_pos <= 0.67
        elif 12 <= g_idx <= 15:    # Gruppe 13 bis 16 (67 bis 84%)
            return 0.67 <= rel_pos <= 0.84
        elif 16 <= g_idx <= 19:    # Gruppe 17 bis 20 (84 bis 100%)
            return 0.84 <= rel_pos <= 1.05
        return True

    i = 0
    for g_idx in range(ERWARTETE_GRUPPEN):
        if gruppe_abgebrochen or i + 2 >= len(peaks_max_idx):
            gruppen.append({'max_idx': np.array([]), 'min_idx': np.array([]), 'aufgeloest': False})
            continue

        ist_valide = False

        if g_idx == 0:
            while i + 2 < len(peaks_max_idx):
                p1, p2, p3 = peaks_max_idx[i], peaks_max_idx[i+1], peaks_max_idx[i+2]
                d1_x = x[p2] - x[p1]
                d2_x = x[p3] - x[p2]

                mittlere_x_pos = x[p2]
                min_abstand = geschätzte_gruppenbreite_x * 0.12
                
                if d1_x >= min_abstand and d2_x >= min_abstand and ist_im_erlaubten_bereich(g_idx, mittlere_x_pos):
                    min1 = finde_echtes_minimum(p1, p2, y_korrigiert)
                    min2 = finde_echtes_minimum(p2, p3, y_korrigiert)

                    gruppen.append({
                        'max_idx': np.array([p1, p2, p3]), 
                        'min_idx': np.array([min1, min2]), 
                        'aufgeloest': True
                    })
                    vorherige_breite_x = x[p3] - x[p1]
                    i += 3
                    ist_valide = True
                    break
                else:
                    i += 1

            if not ist_valide:
                gruppe_abgebrochen = True
                gruppen.append({'max_idx': np.array([]), 'min_idx': np.array([]), 'aufgeloest': False})
            continue

        while i + 2 < len(peaks_max_idx):
            p1, p2, p3 = peaks_max_idx[i], peaks_max_idx[i+1], peaks_max_idx[i+2]

            aktuelle_breite_x = x[p3] - x[p1]
            mittlere_x_pos = x[p2]

            d1 = p2 - p1
            d2 = p3 - p2
            ist_symmetrisch = (d1 / max(d2, 1) <= 2.2) and (d2 / max(d1, 1) <= 2.2)

            obergrenze = vorherige_breite_x + (schrittweite * 1.5)
            untergrenze = vorherige_breite_x * 0.10

            ist_groesse_ok = (aktuelle_breite_x >= untergrenze) and (aktuelle_breite_x <= obergrenze)
            ist_am_richtigen_ort = ist_im_erlaubten_bereich(g_idx, mittlere_x_pos)

            if ist_symmetrisch and ist_groesse_ok and ist_am_richtigen_ort:
                min1 = finde_echtes_minimum(p1, p2, y_korrigiert)
                min2 = finde_echtes_minimum(p2, p3, y_korrigiert)

                gruppen.append({
                    'max_idx': np.array([p1, p2, p3]), 
                    'min_idx': np.array([min1, min2]), 
                    'aufgeloest': True
                })
                vorherige_breite_x = aktuelle_breite_x
                i += 3
                ist_valide = True
                break
            else:
                i += 1

        if not ist_valide:
            gruppe_abgebrochen = True
            gruppen.append({'max_idx': np.array([]), 'min_idx': np.array([]), 'aufgeloest': False})

    while len(gruppen) < ERWARTETE_GRUPPEN:
        gruppen.append({'max_idx': np.array([]), 'min_idx': np.array([]), 'aufgeloest': False})

    # Kontrast- & MTF-Berechnung relativ zur Referenzgruppe C1
    mtf_ergebnisse = []
    c1_referenz = None

    for idx, g in enumerate(gruppen):
        gruppe_name = f"G{idx + 1}"

        if g['aufgeloest']:
            max_werte = y_korrigiert[g['max_idx']]
            min_werte = y_korrigiert[g['min_idx']]

            mittelwert_max = np.mean(max_werte)
            mittelwert_min = np.mean(min_werte) if len(min_werte) > 0 else 0.0

            kontrast = (mittelwert_max - mittelwert_min) / (mittelwert_max + mittelwert_min) if (mittelwert_max + mittelwert_min) > 0 else 0

            if idx == 0:
                c1_referenz = kontrast

            mtf = kontrast / c1_referenz if (c1_referenz and c1_referenz > 0) else 0.0
            mtf = np.clip(mtf, 0, 1)
        else:
            kontrast = 0.0
            mtf = 0.0
            mittelwert_max = 0.0
            mittelwert_min = 0.0

        mtf_ergebnisse.append({
            'Gruppe': gruppe_name,
            'Mittelwert_Max': mittelwert_max,
            'Mittelwert_Min': mittelwert_min,
            'Kontrast': kontrast,
            'MTF': mtf
        })

    return pd.DataFrame(mtf_ergebnisse)


# SCHRITT 2: PROFIL-EXTRAKTION VIA IMAGEJ-MAKRO
def process_single_image(ij, image_path, macro_code, RoiManager, ResultsTable, script_dir, debug_dir):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # ROIs laden & auf Einzelbilder im Stack anwenden
    rm = RoiManager.getRoiManager()
    rm.reset()

    roi_files = glob.glob(os.path.join(script_dir, "*.roi")) + glob.glob(os.path.join(script_dir, "*.zip"))
    if not roi_files:
        print(f"Fehler: Keine .roi oder .zip Datei im Skript-Ordner gefunden!")
        return None

    for roi_path in sorted(roi_files):
        rm.open(roi_path)

    total_rois = rm.getCount()
    if total_rois == 0:
        print("[FEHLER] Keine ROIs im ROI-Manager vorhanden!")
        return None

    imp = ij.IJ.openImage(image_path)
    if imp is None:
        print(f"Fehler: Bild '{base_name}' konnte nicht geladen werden.")
        return None

    imp.show()

    win = imp.getWindow()
    if win is not None:
        win.setLocation(10000, 10000)
    extracted_dfs = {}

    for roi_idx in range(total_rois):
        resolution_num = roi_idx + 1

        rt = ResultsTable.getResultsTable()
        rt.reset()

        rm.select(imp, roi_idx)
        ij.py.run_macro(macro_code)

        rt = ResultsTable.getResultsTable()
        n_rows = rt.getCounter()

        if n_rows == 0:
            continue

        x_vals = [rt.getValue("X", i) for i in range(n_rows)]
        try:
            y_vals = [rt.getValue(f"Grauwert_ROI {resolution_num}", i) for i in range(n_rows)]
        except Exception:
            try:
                y_vals = [rt.getValue("Y", i) for i in range(n_rows)]
            except Exception:
                y_vals = [rt.getValue("Value", i) for i in range(n_rows)]

        extracted_dfs[resolution_num] = pd.DataFrame({"X": x_vals, "Grauwert": y_vals})

    imp.changes = False
    imp.close()

    ij.IJ.run("Close All")

    # Grauwertprofile (ROI 1 & 2) extrahieren & zusammenfügen
    if 1 in extracted_dfs and 2 in extracted_dfs:
        df1 = extracted_dfs[1].copy()
        df2 = extracted_dfs[2].copy()

        x_spalte = "X"
        letzter_x_wert = df1[x_spalte].iloc[-1]
        schrittweite = df1[x_spalte].iloc[1] - df1[x_spalte].iloc[0]

        df2[x_spalte] = df2[x_spalte] + letzter_x_wert + schrittweite
        df_combined = pd.concat([df1, df2], ignore_index=True)

        # AUFRUF SCHRITT 3: Zusammengefügte Profile an die Signalverarbeitung & MTF-Berechnung übergeben
        return calculate_mtf(df_combined, image_name=base_name, debug_dir=debug_dir)
    else:
        print(f"[HINWEIS] Nicht alle ROIs für {base_name} extrahiert. Überspringe...")
        return None


# SCHRITT 1: INITIALISIERUNG & BILDAUSWAHL
if __name__ == "__main__":

    print("\n--------------------------------------------------")
    folder_input = input("Ziehe deine ORDNER mit den Bildern hierher und drücke Enter: ")
    print("--------------------------------------------------")

    folder_paths = [p.strip('"\' ') for p in shlex.split(folder_input)]

    if not folder_paths:
        print("[FEHLER] Keine Ordner angegeben. Abbruch.")
        sys.exit(0)

    # Ordnerpfad & Bildbereich (z.B. 10–75) einlesen
    print("\n--------------------------------------------------")
    user_range = input("Welche Bilder verarbeiten? (z.B. '10-75' oder Enter für alle Bilder pro Ordner): ").strip()
    print("--------------------------------------------------")

    # PyImageJ & Java-Umgebung starten, Makro-Code laden
    print("\nInitialisiere PyImageJ einmalig für alle Ordner...")
    try:
        macro_code = load_macro_code()
        ij = imagej.init(mode='interactive')

        RoiManager = sj.jimport('ij.plugin.frame.RoiManager')
        ResultsTable = sj.jimport('ij.measure.ResultsTable')
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception as e:
        print(f"[SCHWERWIEGENDER FEHLER] PyImageJ konnte nicht initialisiert werden: {e}")
        sys.exit(1)

    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"\n[FEHLER] '{folder_path}' ist kein gültiger Ordner. Überspringe...")
            continue

        print(f"\n---> Verarbeite Ordner: {folder_path}")

        folder_name = os.path.basename(os.path.normpath(folder_path))
        match = re.search(r'\d+', folder_name)
        if match:
            messung_nr = str(int(match.group(0)))
        else:
            messung_nr = folder_name

        ergebnisse_dir = os.path.join(folder_path, "Ergebnisse")
        debug_dir = os.path.join(ergebnisse_dir, "Diagnose_Plots")
        os.makedirs(debug_dir, exist_ok=True)

        valid_extensions = ("*.tif", "*.tiff", "*.dcm", "*.jpg", "*.jpeg", "*.png")
        image_files = []
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

        image_files = sorted(list(set(image_files)))
        total_found = len(image_files)

        if total_found == 0:
            print("[FEHLER] Keine passenden Bilddateien im Ordner gefunden! Überspringe...")
            continue

        start_idx = 1
        end_idx = total_found

        if user_range:
            try:
                if "-" in user_range:
                    parts = user_range.split("-")
                    start_idx = int(parts[0].strip())
                    end_idx = int(parts[1].strip())
                else:
                    start_idx = int(user_range)
                    end_idx = start_idx
            except (ValueError, IndexError):
                print("[HINWEIS] Ungültiges Format. Verwende alle Bilder.")
                start_idx, end_idx = 1, total_found

        start_idx = max(1, min(start_idx, total_found)) - 1
        end_idx = max(start_idx + 1, min(end_idx, total_found))

        selected_files = image_files[start_idx:end_idx]
        print(f"--> Es werden {len(selected_files)} Bilder verarbeitet (Bild {start_idx+1} bis {end_idx} von {total_found}).")

        try:
            all_mtf_results = []
            ortsfrequenzen = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 5.0]

            for index, file_path in enumerate(selected_files, start=1):
                b_name = os.path.basename(file_path)
                print(f"[{index}/{len(selected_files)}] Verarbeite: {b_name}")

                # AUFRUF SCHRITT 2: Profil-Extraktion via ImageJ für das aktuelle Bild starten
                df_mtf_single = process_single_image(ij, file_path, macro_code, RoiManager, ResultsTable, script_dir, debug_dir)

                if df_mtf_single is not None:
                    all_mtf_results.append(df_mtf_single['MTF'].to_numpy())

            # SCHRITT 4: SPLINE-INTERPOLATION & KENNZAHLEN
            if all_mtf_results:
                print("\n--------------------------------------------------")
                print(" BERECHNE KENNZAHLEN (SPLINE-INTERPOLATION)")
                print("--------------------------------------------------")

                x_freqs_orig = np.array(ortsfrequenzen[:len(all_mtf_results[0])])
                x_min_real = min(x_freqs_orig)
                x_fit = np.linspace(x_min_real, max(x_freqs_orig), 500)
                x_search = np.linspace(x_min_real, max(x_freqs_orig) * 2, 2000)

                trap_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
                n_images = len(all_mtf_results)

                single_curves_abs = []
                single_aucs_abs = []
                single_mtf50s = []
                single_mtf10s = []
                single_x_zeros = []

                # Pchip-Spline-Interpolation der MTF-Kurve
                for single_mtf in all_mtf_results:
                    cs = PchipInterpolator(x_freqs_orig, single_mtf)
                    
                    y_single_abs = np.clip(cs(x_fit), 0, 1)
                    single_curves_abs.append(y_single_abs)
                    
                    # Integrale (AUC) berechnen
                    single_aucs_abs.append(abs(trap_func(y_single_abs, x_fit)))
                    
                    # Kennfrequenzen (MTF50, MTF10) ermitteln
                    single_mtf50s.append(x_fit[np.argmin(np.abs(y_single_abs - 0.50))])
                    single_mtf10s.append(x_fit[np.argmin(np.abs(y_single_abs - 0.10))])

                    y_s_search = np.clip(cs(x_search), 0, 1)
                    z_c = x_search[y_s_search <= 0.001]
                    xz_s = z_c[0] if len(z_c) > 0 else max(x_freqs_orig)
                    single_x_zeros.append(xz_s)

                mean_x_zero = np.mean(single_x_zeros)

                x_norm_start = x_min_real / mean_x_zero
                x_fit_norm_common = np.linspace(x_norm_start, 1.0, 500)

                single_curves_norm = []
                single_aucs_norm = []

                for single_mtf, xz_s in zip(all_mtf_results, single_x_zeros):
                    cs = PchipInterpolator(x_freqs_orig, single_mtf)

                    x_norm_single_start = x_min_real / xz_s
                    x_fit_norm_single = np.linspace(x_norm_single_start, 1.0, 500)
                    y_single_norm = np.clip(cs(x_fit_norm_single * xz_s), 0, 1)
                    
                    single_aucs_norm.append(abs(trap_func(y_single_norm, x_fit_norm_single)))

                    y_common_norm = np.clip(cs(x_fit_norm_common * mean_x_zero), 0, 1)
                    single_curves_norm.append(y_common_norm)

                mean_curve_abs = np.mean(single_curves_abs, axis=0)
                sd_curve_abs = np.std(single_curves_abs, axis=0, ddof=1)

                mean_curve_norm = np.mean(single_curves_norm, axis=0)
                sd_curve_norm = np.std(single_curves_norm, axis=0, ddof=1)

                mean_auc_abs = np.mean(single_aucs_abs)
                sd_auc_abs = np.std(single_aucs_abs, ddof=1)
                sem_auc_abs = sd_auc_abs / np.sqrt(n_images)

                mean_auc_norm = np.mean(single_aucs_norm)
                sd_auc_norm = np.std(single_aucs_norm, ddof=1)
                sem_auc_norm = sd_auc_norm / np.sqrt(n_images)

                mean_50 = np.mean(single_mtf50s)
                sd_50 = np.std(single_mtf50s, ddof=1)

                mean_10 = np.mean(single_mtf10s)
                sd_10 = np.std(single_mtf10s, ddof=1)

                mean_50_norm = mean_50 / mean_x_zero
                sd_50_norm = sd_50 / mean_x_zero

                mean_10_norm = mean_10 / mean_x_zero
                sd_10_norm = sd_10 / mean_x_zero

                print("\n--------------------------------------------------")
                print("         ERGEBNISSE UND FEHLERANALYSE")
                print("--------------------------------------------------")
                print("Methode: Pchip-Spline-Interpolation (Monotonieerhaltend)")
                print(f"Messbereich-Start: {x_min_real:.2f} Lp/mm  -->  Normierter Start bei ~{x_norm_start:.3f}")
                print(f"Mittlere erste Nullstelle (y=0): {mean_x_zero:.3f} Lp/mm\n")
                print("--- UNNORMERTES INTEGRAL (AUC Absolut) ---")
                print(f"  AUC: {mean_auc_abs:.3f} ± {sd_auc_abs:.3f} (SD) | ± {sem_auc_abs:.3f} (SEM)")

                print("\n--- NORMIERTES INTEGRAL (AUC Normiert im Messbereich) ---")
                print(f"  AUC: {mean_auc_norm:.4f} ± {sd_auc_norm:.4f} (SD) | ± {sem_auc_norm:.4f} (SEM)")
                
                print("\n--- KENNFREQUENZEN ---")
                print(f"  MTF50: {mean_50:.3f} ± {sd_50:.3f} Lp/mm")
                print(f"  MTF10: {mean_10:.3f} ± {sd_10:.3f} Lp/mm")
                print("--------------------------------------------------\n")

                # SCHRITT 5: EXPORT
                # CSV-Dateien speichern (Unnormierte & Normierte Daten)
                pd.DataFrame({
                    'Ortsfrequenz_Lp_mm': x_fit,
                    'MTF_Mittelwert': mean_curve_abs,
                    'MTF_SD': sd_curve_abs,
                }).to_csv(os.path.join(ergebnisse_dir, "Gesamt_MTF_Unnormiert.csv"), sep=";", decimal=",", index=False, encoding="utf-8-sig")

                pd.DataFrame({
                    'Ortsfrequenz_Normiert': x_fit_norm_common,
                    'Ortsfrequenz_Reale_Lp_mm_Mittelwert': x_fit_norm_common * mean_x_zero,
                    'MTF_Mittelwert_Normiert': mean_curve_norm,
                    'MTF_SD_Normiert': sd_curve_norm,
                }).to_csv(os.path.join(ergebnisse_dir, "Gesamt_MTF_Normiert.csv"), sep=";", decimal=",", index=False, encoding="utf-8-sig")

                pd.DataFrame([{
                    'Methode': "Pchip-Spline-Interpolation",
                    'Messbereich_Start_Lp_mm': x_min_real,
                    'Nullstelle_y0_Lp_mm_Mittelwert': mean_x_zero,
                    'AUC_Absolut': mean_auc_abs, 'AUC_Absolut_SD': sd_auc_abs, 'AUC_Absolut_SEM': sem_auc_abs,
                    'AUC_Normiert': mean_auc_norm, 'AUC_Normiert_SD': sd_auc_norm, 'AUC_Normiert_SEM': sem_auc_norm,
                    'MTF50_Lp_mm': mean_50, 'MTF50_SD': sd_50,
                    'MTF10_Lp_mm': mean_10, 'MTF10_SD': sd_10
                }]).to_csv(os.path.join(ergebnisse_dir, "MTF_Kennzahlen.csv"), sep=";", decimal=",", index=False, encoding="utf-8-sig")

                # Hochauflösende MTF-Diagramme (.png) mit Fehlerbalken generieren
                fig1, ax1 = plt.subplots(figsize=(11, 6))

                for i, single_mtf in enumerate(all_mtf_results):
                    ax1.scatter(x_freqs_orig, single_mtf, color='#1f77b4', alpha=0.30, s=18, edgecolors='none', label="Einzelbild-Punkte" if i == 0 else "")

                ax1.plot(x_fit, mean_curve_abs, color='#d62728', linewidth=2.8, zorder=10, label=f"Mittlere MTF-Kurve (N={n_images})")

                ax1.axhline(y=0.50, color='navy', linestyle=':', linewidth=1.2, zorder=3)
                ax1.axvline(x=mean_50, color='navy', linestyle='--', linewidth=1.5, zorder=3)
                ax1.errorbar(x=mean_50, y=0.50, xerr=sd_50, fmt='o', color='navy', ecolor='navy', elinewidth=2.0, capsize=4, markersize=7, zorder=15, label=f"MTF50 ({mean_50:.2f} ± {sd_50:.2f} Lp/mm)")
                
                ax1.axhline(y=0.10, color='green', linestyle=':', linewidth=1.2, zorder=3)
                ax1.axvline(x=mean_10, color='green', linestyle='--', linewidth=1.5, zorder=3)
                ax1.errorbar(x=mean_10, y=0.10, xerr=sd_10, fmt='o', color='green', ecolor='darkgreen', elinewidth=2.0, capsize=4, markersize=7, zorder=15, label=f"MTF10 ({mean_10:.2f} ± {sd_10:.2f} Lp/mm)")

                ax1.text(mean_50 + 0.05, 0.53, f"MTF50: {mean_50:.2f} Lp/mm", color='navy', fontweight='bold', fontsize=9.5, zorder=20)
                ax1.text(mean_10 + 0.05, 0.13, f"MTF10: {mean_10:.2f} Lp/mm", color='darkgreen', fontweight='bold', fontsize=9.5, zorder=20)

                info_abs = (
                    f"Method: Spline-Interpolation\n"
                    f"AUC (Absolut): {mean_auc_abs:.3f} ± {sd_auc_abs:.3f}\n"
                    f"MTF50: {mean_50:.2f} ± {sd_50:.2f} Lp/mm\n"
                    f"MTF10: {mean_10:.2f} ± {sd_10:.2f} Lp/mm\n"
                    f"Nullstelle (y=0): {mean_x_zero:.2f} Lp/mm"
                )
                ax1.text(0.97, 0.65, info_abs, transform=ax1.transAxes, fontsize=9.0, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff7bc", edgecolor="#d95f0e", alpha=0.95))

                ax1.set_xlabel("Ortsfrequenz [Lp/mm]", fontsize=11, fontweight='bold')
                ax1.set_ylabel("MTF", fontsize=11, fontweight='bold')
                ax1.set_title(f"Unnormierte MTF-Kurve – Messung {messung_nr}", fontsize=12, fontweight='bold')
                ax1.set_ylim(-0.05, 1.05)
                ax1.set_xlim(min(x_freqs_orig), max(x_freqs_orig) + 0.2)
                ax1.grid(True, linestyle='--', alpha=0.6)
                ax1.legend(loc="upper right", frameon=True)
                plt.tight_layout()

                path_plot_abs = os.path.join(ergebnisse_dir, "Gesamt_MTF_Unnormiert.png")
                plt.savefig(path_plot_abs, dpi=300, bbox_inches="tight")
                plt.close(fig1)

                fig2, ax2 = plt.subplots(figsize=(11, 6))

                for i, (single_mtf, xz) in enumerate(zip(all_mtf_results, single_x_zeros)):
                    ax2.scatter(x_freqs_orig / xz, single_mtf, color='#1f77b4', alpha=0.30, s=18, edgecolors='none', label="Einzelbild-Punkte (normiert)" if i == 0 else "")

                ax2.plot(x_fit_norm_common, mean_curve_norm, color='#2ca02c', linewidth=2.8, zorder=10, label=f"Mittlere Normierte MTF-Kurve ({x_norm_start:.2f} bis 1.0)")

                ax2.axhline(y=0.50, color='navy', linestyle=':', linewidth=1.2, zorder=3)
                ax2.axvline(x=mean_50_norm, color='navy', linestyle='--', linewidth=1.5, zorder=3)
                ax2.errorbar(x=mean_50_norm, y=0.50, xerr=sd_50_norm, fmt='o', color='navy', ecolor='navy', elinewidth=2.0, capsize=4, markersize=7, zorder=15, label=f"MTF50 ({mean_50:.2f} Lp/mm ≙ {mean_50_norm:.2f})")
                
                ax2.axhline(y=0.10, color='green', linestyle=':', linewidth=1.2, zorder=3)
                ax2.axvline(x=mean_10_norm, color='green', linestyle='--', linewidth=1.5, zorder=3)
                ax2.errorbar(x=mean_10_norm, y=0.10, xerr=sd_10_norm, fmt='o', color='green', ecolor='darkgreen', elinewidth=2.0, capsize=4, markersize=7, zorder=15, label=f"MTF10 ({mean_10:.2f} Lp/mm ≙ {mean_10_norm:.2f})")

                ax2.text(mean_50_norm + 0.02, 0.53, f"MTF50: {mean_50:.2f} Lp/mm", color='navy', fontweight='bold', fontsize=9.5, zorder=20)
                ax2.text(mean_10_norm + 0.02, 0.13, f"MTF10: {mean_10:.2f} Lp/mm", color='darkgreen', fontweight='bold', fontsize=9.5, zorder=20)

                info_norm = (
                    f"AUC (Normiert): {mean_auc_norm:.4f} ± {sd_auc_norm:.4f}\n"
                    f"MTF50: {mean_50:.2f} Lp/mm (norm: {mean_50_norm:.2f})\n"
                    f"MTF10: {mean_10:.2f} Lp/mm (norm: {mean_10_norm:.2f})\n"
                    f"Nullstelle (y=0): {mean_x_zero:.2f} Lp/mm"
                )
                ax2.text(0.97, 0.65, info_norm, transform=ax2.transAxes, fontsize=9.5, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="#e5f5e0", edgecolor="#31a354", alpha=0.95))

                ax2.set_xlabel("Normierte Ortsfrequenz ", fontsize=11, fontweight='bold')
                ax2.set_ylabel("MTF", fontsize=11, fontweight='bold')
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_xlim(x_norm_start - 0.05, 1.05)
                ax2.grid(True, linestyle='--', alpha=0.6)
                ax2.legend(loc="upper right", frameon=True)
                plt.tight_layout()

                path_plot_norm = os.path.join(ergebnisse_dir, "Gesamt_MTF_Normiert.png")
                plt.savefig(path_plot_norm, dpi=300, bbox_inches="tight")
                plt.close(fig2)

                print(f"[ERFOLG] Graphen und Daten erfolgreich gespeichert:")
                print(f"  -> PNG (Unnormiert): {path_plot_abs}")
                print(f"  -> PNG (Normiert):   {path_plot_norm}")

            print("\n--> Räume temporäre Dateien auf...")
            ij.IJ.run("Close All")

            temp_csvs = glob.glob(os.path.join(folder_path, "*_Auflösung-*.csv"))
            for t_csv in temp_csvs:
                try:
                    os.remove(t_csv)
                except Exception:
                    pass

        except Exception as e:
            print(f"\nFehler beim Verarbeiten von {folder_path}: {e}")

    print("\n--------------------------------------------------")
    print(" FERTIG! Alle Ordner wurden verarbeitet.")
    print("--------------------------------------------------")

    plt.close('all')

    try:
        input("\nDrücke Enter zum Beenden...")
    except KeyboardInterrupt:
        pass

    os._exit(0)