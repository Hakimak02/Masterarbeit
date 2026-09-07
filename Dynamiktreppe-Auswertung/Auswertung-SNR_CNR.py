"""
Autor: Hakim Kayed
Verwendungszweck: Masterarbeit zur Untersuchung der Bildqualität von Angiografiebildern
Hinweis / Zitat: Erstellt, übersetzt und optimiert unter Nutzung von Google Gemini 3.1 Pro 
                 zur Übersetzung von Konzepten/Ideen in Python-Code sowie zur Code-Optimierung.
"""

import os
import sys
import glob
import re
import shlex
import imagej
import scyjava as sj

# SCHRITT 1: INITIALISIERUNG & EINGABE

# Java-Laufzeitumgebung konfigurieren
sj.config.set_java_constraints(version="21", vendor="temurin")

def load_macro_code():
    """
    SCHRITT 1: Makro 'Macro-SNR_CNR.ijm' laden
    Liest den Makro-Code aus der Datei im selben Verzeichnis.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    macro_path = os.path.join(script_dir, "Macro-SNR_CNR.ijm")
    
    if not os.path.exists(macro_path):
        raise FileNotFoundError(f"Makro-Datei nicht gefunden unter: {macro_path}")
        
    with open(macro_path, "r", encoding="utf-8") as f:
        return f.read()

def process_single_folder(ij, images_dir, macro_code, RoiManager, FolderOpener):
    """Verarbeitet einen einzelnen Ordner vollautomatisch"""
    images_dir = images_dir.strip('"\' ')
    if not os.path.isdir(images_dir):
        print(f"\n[WARNUNG] '{images_dir}' ist kein gültiger Ordner. Überspringe...")
        return

    print(f"\n Starte Analyse für: {os.path.basename(images_dir)}")

    # SCHRITT 2: ROI-VERWALTUNG (JE ORDNER)
    
    # RoiManager zurücksetzen
    rm = RoiManager.getRoiManager()
    rm.reset()

    # Im übergeordneten Ordner nach ROIs suchen
    parent_dir = os.path.abspath(os.path.join(images_dir, ".."))
    roi_paths = []

    # 'ROI_Background_Noise.roi' laden
    bg_roi = os.path.join(parent_dir, "ROI_Background_Noise.roi")
    if os.path.exists(bg_roi):
        roi_paths.append(bg_roi)
    else:
        print(f"Warnung: Background-ROI nicht gefunden unter {bg_roi}")

    # 'ROI_Step_*.roi' numerisch sortiert öffnen
    step_rois = glob.glob(os.path.join(parent_dir, "ROI_Step_*.roi"))
    
    # Komplexer Funktionsblock: Extrahiert die Zahl aus dem Dateinamen mittels RegEx,
    # um die Stufen-ROIs korrekt numerisch zu sortieren (z.B. Step_2 vor Step_10).
    def extract_step_number(filepath):
        match = re.search(r'ROI_Step_(\d+)\.roi', os.path.basename(filepath))
        return int(match.group(1)) if match else 0

    step_rois = sorted(step_rois, key=extract_step_number)
    roi_paths.extend(step_rois)

    if not roi_paths:
        print(f"Fehler: Keine ROIs in {parent_dir} gefunden!")
        return

    for roi_path in roi_paths:
        rm.runCommand("Open", roi_path)

    # SCHRITT 3: BILDSTACK & MAKRO-AUSFÜHRUNG
    
    # Bildstack via FolderOpener laden & anzeigen
    print(f"Öffne Bildstack aus: {images_dir}")
    imp = FolderOpener.open(images_dir)
    
    if imp is None:
        print(f"Fehler: Stack konnte nicht geladen werden.")
        return

    imp.show()

    # ImageJ-Makro 'Macro-SNR_CNR.ijm' ausführen (SNR & CNR Berechnungen durchführen)
    ij.py.run_macro(macro_code)

    # SCHRITT 4: ABSCHLUSS & REINIGUNG
    
    # Bildstack ohne Speichern schließen (`changes=False`)
    imp.changes = False
    imp.close()
    print(f"--> Fertig mit: {os.path.basename(images_dir)}")


if __name__ == "__main__":
    # SCHRITT 1: Ordnerpfade über Konsole einlesen
    user_input = input("\nZiehe deine(n) Bildordner hierher und drücke Enter: ")

    if not user_input.strip():
        print("Keine Eingabe erhalten. Abbruch.")
        sys.exit(0)

    # Komplexer Funktionsblock: Nutzt `shlex.split`, um durch Drag-and-Drop 
    # hintereinander gezogene Ordnerpfade korrekt zu trennen (inkl. Anführungszeichen/Leerzeichen).
    try:
        folders = shlex.split(user_input, posix=False)
    except Exception:
        folders = user_input.split()

    print(f"\n-> {len(folders)} Ordner erkannt. Initialisiere PyImageJ...")

    try:
        macro_code = load_macro_code()

        # SCHRITT 1: PyImageJ Gateway im 'interactive'-Modus starten
        ij = imagej.init(mode='interactive')
        
        # SCHRITT 1: Java-Klassen (RoiManager, FolderOpener) importieren
        RoiManager = sj.jimport('ij.plugin.frame.RoiManager')
        FolderOpener = sj.jimport('ij.plugin.FolderOpener')

        # SCHRITT 4: Nächsten Ordner verarbeiten (Schleife über alle Ordner)
        for index, folder in enumerate(folders, start=1):
            print(f"\nFortschritt: Ordner [{index}/{len(folders)}]")
            process_single_folder(ij, folder, macro_code, RoiManager, FolderOpener)

        print("\n ALLE ORDNER WURDEN ERFOLGREICH VERARBEITET!")

    except Exception as e:
        print(f"\nFehler aufgetreten: {e}")

    # SCHRITT 4: Beendung nach Benutzereingabe
    input("\nDrücke Enter zum Beenden...")