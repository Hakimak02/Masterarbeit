"""
Autor: Hakim Kayed
Verwendungszweck: Masterarbeit zur Untersuchung der Bildqualität von Angiografiebildern
Hinweis / Zitat: Erstellt, übersetzt und optimiert unter Nutzung von Google Gemini 3.1 Pro 
                 zur Übersetzung von Konzepten/Ideen in Python-Code sowie zur Code-Optimierung.
"""

import os
import sys
import glob
import cv2
import numpy as np
import roifile

# Ordner bestimmen, in dem DIESES Python-Skript liegt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_folder_path():
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Ziehe deinen Bildordner hierher und drücke Enter: ")
    return folder_path.strip('"\'').rstrip('/\\')


# SCHRITT 1: BILDVORVERARBEITUNG
def create_average_image(folder_path):
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    # Bilder einlesen & erste 10 überspringen
    image_paths = sorted(list(set(image_paths)))
    if len(image_paths) > 10:
        image_paths = image_paths[10:]

    if not image_paths:
        raise ValueError(f"Keine unterstützten Bilder im Ordner gefunden: {folder_path}")

    first_img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    if first_img is None:
        raise ValueError(f"Bild konnte nicht gelesen werden: {image_paths[0]}")
    
    if len(first_img.shape) == 3:
        first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)

    stack = np.zeros((len(image_paths), *first_img.shape), dtype=np.float64)

    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stack[i] = img

    # Pixelweise Mittelung (np.mean) & 8-Bit-Normierung
    avg_raw = np.mean(stack, axis=0)
    avg_normalized = cv2.normalize(avg_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return avg_normalized, len(image_paths)


# SCHRITT 2: BILDVERBESSUNG & MUSTERSUCHE
def detect_pattern_directly_from_enhanced(avg_img):
    img_h, img_w = avg_img.shape
    
    # Bild-Zuschnitt zur Randeliminierung (Äußerer Rahmen & Störobjekte unten entfernen)
    pad_top = int(img_h * 0.07)
    pad_bottom = int(img_h * 0.75)  # Verwirft die unteren 25% des Bildes
    pad_left = int(img_w * 0.07)
    pad_right = int(img_w * 0.93)

    cropped_img = avg_img[pad_top:pad_bottom, pad_left:pad_right]
    crop_h, crop_w = cropped_img.shape
    crop_area = crop_h * crop_w

    # Kontrastverstärkung (CLAHE) & Otsu-Binarisierung
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cropped_img)
    cv2.imwrite(os.path.join(SCRIPT_DIR, "debug_01_enhanced.png"), enhanced)

    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    max_area = 0

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        
        # Flächenprüfung relativ zum zugeschnittenen Bildbereich
        if crop_area * 0.05 < area < crop_area * 0.60:
            rect = cv2.minAreaRect(hull)
            (cx, cy), (w, h), angle = rect
            
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio < 1.6 and area > max_area:
                max_area = area
                best_rect = rect

    if best_rect is None:
        raise ValueError("Konnte das Blei-Plättchen nicht isolieren.")

    # Bounding Box Koordinaten (minAreaRect) auf Originalbild zurückrechnen
    (cx, cy), (w, h), angle = best_rect
    orig_center = (cx + pad_left, cy + pad_top)
    orig_rect = (orig_center, (w, h), angle)

    return orig_rect


# SCHRITT 3: KOORDINATEN-TRANSFORMATION & ROIs
def compute_rois_diagonal_split(rect):
    """
    Richtet die ROIs STRIKT so aus, dass die Trennlinie immer waagerecht/quer 
    von der linken zur rechten Diagonale verläuft.
    """
    box = cv2.boxPoints(rect)
    
    # Die 4 Extrempunkte der Raute im Bild ermitteln
    p_top = box[np.argmin(box[:, 1])]     # Höchster Punkt
    p_bottom = box[np.argmax(box[:, 1])]  # Tiefster Punkt
    p_left = box[np.argmin(box[:, 0])]    # Linkester Punkt
    p_right = box[np.argmax(box[:, 0])]   # Rechester Punkt

    p0 = p_top
    p1 = p_right
    p2 = p_bottom
    p3 = p_left

    margin_w = 0.09  # Rand zu den Seiten (9%)
    margin_h = 0.15  # Rand oben/unten (15%)

    # Bilineare Interpolation zur Ausrichtung der Teil-ROIs mit Margins
    def interp(u, v):
        top = p0 + u * (p1 - p0)
        bottom = p3 + u * (p2 - p3)
        return top + v * (bottom - top)

    # 2 Teil-ROIs (0.6–1.6 & 1.8–5.0 LP) ausrichten
    r1 = np.array([
        interp(margin_w, margin_h),
        interp(1.0 - margin_w, margin_h),
        interp(1.0 - margin_w, 0.50),
        interp(margin_w, 0.50)
    ])

    r2 = np.array([
        interp(margin_w, 0.50),
        interp(1.0 - margin_w, 0.50),
        interp(1.0 - margin_w, 1.0 - margin_h),
        interp(margin_w, 1.0 - margin_h)
    ])

    outer_pts = np.array([p0, p1, p2, p3])
    return r1, r2, outer_pts


# SCHRITT 4: EXPORT & VORSCHAU
def save_rotated_rect_roi(points, name):
    pts = np.array(points, dtype=np.float32)
    
    x1, y1 = (pts[0] + pts[3]) / 2.0
    x2, y2 = (pts[1] + pts[2]) / 2.0
    
    top_mid = (pts[0] + pts[1]) / 2.0
    bottom_mid = (pts[3] + pts[2]) / 2.0
    
    raw_stroke_width = np.linalg.norm(top_mid - bottom_mid)
    stroke_width = raw_stroke_width * 0.75  

    # ImageJ-Linien-ROIs erzeugen & als .roi speichern
    roi = roifile.ImagejRoi()
    roi.roitype = roifile.ROI_TYPE.LINE
    
    roi.x1, roi.y1 = float(x1), float(y1)
    roi.x2, roi.y2 = float(x2), float(y2)
    
    roi.left = int(min(x1, x2))
    roi.top = int(min(y1, y2))
    roi.right = int(max(x1, x2))
    roi.bottom = int(max(y1, y2))
    
    roi.stroke_width = int(round(stroke_width))
    roi.name = name
    
    out_path = os.path.join(SCRIPT_DIR, f"{name}.roi")
    roi.tofile(out_path)
    print(f"ROI gespeichert mit Dicke {roi.stroke_width}: {out_path}")


def process_linepair_rois(avg_img):
    rect = detect_pattern_directly_from_enhanced(avg_img)
    
    # Ort der Rauten-Hälften ermitteln (Schritt 3)
    roi1_pts, roi2_pts, outer_pts = compute_rois_diagonal_split(rect)

    # ROIs speichern (Schritt 4)
    save_rotated_rect_roi(roi1_pts, "ROI_Resolution_0.6-1.6")
    save_rotated_rect_roi(roi2_pts, "ROI_Resolution_1.8-5.0")

    # Vorschau 'debug_preview_resolution.png' abspeichern (Schritt 4)
    preview = cv2.cvtColor(avg_img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(preview, [np.int32(outer_pts)], True, (0, 0, 255), 2)
    cv2.polylines(preview, [np.int32(roi1_pts)], True, (255, 0, 255), 2)
    cv2.polylines(preview, [np.int32(roi2_pts)], True, (255, 0, 255), 2)

    debug_path = os.path.join(SCRIPT_DIR, "debug_preview_resolution.png")
    cv2.imwrite(debug_path, preview)
    print(f"-> Finale Vorschau gespeichert in: {debug_path}")


if __name__ == "__main__":
    try:
        folder = get_folder_path()
        print(f"\nVerarbeite Ordner: {folder}")
        print(f"Zielordner für Ergebnisse: {SCRIPT_DIR}\n")
        
        # Ablauf starten
        avg_img, count = create_average_image(folder)
        print(f"{count} Bilder erfolgreich gemittelt.")

        cv2.imwrite(os.path.join(SCRIPT_DIR, "debug_00_averaged.png"), avg_img)

        process_linepair_rois(avg_img)
        print("\nROIs erfolgreich erzeugt!")

    except Exception as e:
        print(f"\nHinweis/Fehler: {e}")

    input("\nDrücke Enter zum Beenden...")