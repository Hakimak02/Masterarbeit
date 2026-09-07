import os
import sys
import glob
import re
import cv2
import numpy as np
import roifile


# SCHRITT 1: BENUTZEREINGABEN & ORDNERSUCHE


# 1.1 Pfad zum Bildordner ermitteln
def get_folder_path():
    if len(sys.argv) > 1:
        folder_path = sys.argv[1] 
    else:
        folder_path = input("Ziehe deinen Bildordner hierher und drücke Enter: ")
    return folder_path.strip('"\'')

# 1.2 Abfrage der sichtbaren Stufenbereichs
def get_step_range():
    print("\nWelche Stufen der Dynamiktreppe sind auf dem Bild zu sehen?")
    user_input = input("Eingabe (z. B. '1-7' oder '3-7'): ").strip()
    
    numbers = re.findall(r'\d+', user_input)
    if len(numbers) == 1:
        start_step = end_step = int(numbers[0])
    elif len(numbers) >= 2:
        start_step, end_step = int(numbers[0]), int(numbers[1])
    else:
        print("Standard verwendet: Stufen 1 bis 7")
        return 1, 7

    start_step = max(1, min(7, start_step))
    end_step = max(1, min(7, end_step))
    
    if start_step > end_step:
        start_step, end_step = end_step, start_step
        
    print(f"-> Verarbeite Stufen {start_step} bis {end_step}.")
    return start_step, end_step



# SCHRITT 2: BILD-VORVERARBEITUNG & MITTELWERTBILD


def create_average_image(folder_path):
    """Lädt alle Bilddateien, überspringt die ersten 10 Bilder und berechnet ein Mittelwertbild."""
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        raise ValueError(f"Keine unterstützten Bilder im Ordner gefunden: {folder_path}")

    # 2.1 Einschwingphase überspringen (Erste 10 Slices aussortieren)
    if len(image_paths) > 10:
        image_paths = image_paths[10:]

    # 2.2 Erstes Bild einlesen und Bilddimensionen festlegen
    first_img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    if first_img is None:
        raise ValueError(f"Bild konnte nicht gelesen werden: {image_paths[0]}")
    
    if len(first_img.shape) == 3:
        first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)

    stack = np.zeros((len(image_paths), *first_img.shape), dtype=np.float64)

    # 2.3 Bildstack laden & Grauwert-Mittelwert berechnen
    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stack[i] = img

    # 2.4 Normalisierung auf 8-Bit (0-255)
    avg_raw = np.mean(stack, axis=0)
    avg_normalized = cv2.normalize(avg_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return avg_normalized, len(image_paths)



# SCHRITT 3: BILDANALYSE & STUFENERKENNUNG (COMPUTER VISION)


# 3.1 Aktiven Detektorbereich ermitteln (Randartefakte abschneiden)
def get_active_detector_area(img):
    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY) 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        margin_x = int(w * 0.02)
        margin_y = int(h * 0.02)
        return x + margin_x, y + margin_y, w - 2 * margin_x, h - 2 * margin_y
    return 0, 0, img.shape[1], img.shape[0]

# 3.2 Geometrie der Dynamiktreppe automatisch lokalisieren
def detect_exact_steps(avg_img, num_steps_visible):
    img_h, img_w = avg_img.shape
    ax, ay, aw, ah = get_active_detector_area(avg_img)

    roi_img = avg_img[ay:ay+ah, ax:ax+aw]

    # --- Step 3.2.1: Kontrastverstärkung & Filterung ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi_img)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # --- Step 3.2.2: Vertikale Kanten suchen (Treppenbreite X) ---
    sobel_x = np.abs(cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3))
    col_energy = np.mean(sobel_x, axis=0)
    col_energy_smooth = cv2.GaussianBlur(col_energy.reshape(1, -1), (15, 1), 0).ravel()

    threshold_x = np.max(col_energy_smooth) * 0.20
    peaks_x = []
    for x in range(1, aw - 1):
        if (col_energy_smooth[x] > threshold_x and 
            col_energy_smooth[x] >= col_energy_smooth[x-1] and 
            col_energy_smooth[x] >= col_energy_smooth[x+1]):
            peaks_x.append(x)

    valid_peaks_x = [p for p in peaks_x if int(aw * 0.05) < p < int(aw * 0.95)]
    valid_peaks_x.sort(key=lambda p: col_energy_smooth[p], reverse=True)

    if len(valid_peaks_x) >= 2:
        p1 = valid_peaks_x[0]
        p2 = next((p for p in valid_peaks_x[1:] if abs(p - p1) >= int(aw * 0.05)), None)
        if p2 is not None:
            x_left = min(p1, p2)
            x_right = max(p1, p2)
        else:
            x_left = p1
            x_right = aw
    elif len(valid_peaks_x) == 1:
        x_left = valid_peaks_x[0]
        x_right = aw
    else:
        x_left = int(aw * 0.6)
        x_right = int(aw * 0.95)

    w_p = x_right - x_left
    global_x_left = ax + x_left
    global_w_p = w_p

    # --- Step 3.2.3: Horizontale Kanten & Stufenhöhe finden (Y-Profil) ---
    strip_x1 = x_left + int(w_p * 0.20)
    strip_x2 = min(aw - 2, x_right - int(w_p * 0.20))
    if strip_x2 <= strip_x1:
        strip_x2 = strip_x1 + 5

    profile = np.mean(roi_img[:, strip_x1:strip_x2], axis=1)

    grad = np.abs(np.gradient(profile))
    grad_smooth = cv2.GaussianBlur(grad.reshape(-1, 1), (1, 5), 0).ravel()

    pos_thresh = np.max(grad_smooth) * 0.15
    peaks_y = []
    for y in range(1, ah - 1):
        if (grad_smooth[y] > pos_thresh and 
            grad_smooth[y] >= grad_smooth[y-1] and 
            grad_smooth[y] >= grad_smooth[y+1]):
            peaks_y.append(y)

    min_step_h = int(ah * 0.08)
    filtered_peaks = []
    for py in peaks_y:
        if not filtered_peaks or (py - filtered_peaks[-1] >= min_step_h):
            filtered_peaks.append(py)

    if len(filtered_peaks) > 1:
        step_h = int(np.median(np.diff(filtered_peaks)))
    else:
        step_h = int(ah / (num_steps_visible + 1))

    if filtered_peaks:
        top_edge = filtered_peaks[0]
    else:
        top_edge = int(ah * 0.1)

    # --- Step 3.2.4: Zuordnung der einzelnen Stufen-Rechtecke ---
    steps = {}
    for i in range(num_steps_visible):
        sy = ay + top_edge + i * step_h
        steps[i] = (global_x_left, sy, global_w_p, step_h)

    return steps, global_x_left



# SCHRITT 4: ROI-GENERIERUNG, BERECHNUNG & EXPORT


# 4.1 Export-Funktion für ImageJ (.roi Dateien)
def save_roi(a, b, c, d, name, output_folder):
    points = [[a, b], [c, b], [c, d], [a, d]]
    roi = roifile.ImagejRoi.frompoints(points)
    roi.roitype = roifile.ROI_TYPE.RECT
    roi.name = name
    
    out_path = os.path.join(output_folder, "..", f"{name}.roi")
    roi.tofile(out_path)
    print(f"ROI gespeichert: {out_path} -> [a={a}, b={b}, c={c}, d={d}]")

# 4.2 Geometrische Berechnung aller ROIs & Erstellen der Artefakte
def process_rois(avg_img, folder, start_step, end_step):
    img_h, img_w = avg_img.shape
    num_visible = end_step - start_step + 1
    steps_map, x_left = detect_exact_steps(avg_img, num_visible)

    # --- Step 4.2.1: Randeinschub (20% Inset Margin) berechnen ---
    def get_inset_roi(step_rect):
        sx, sy, sw, sh = step_rect
        pad_x = int(sw * 0.20)
        pad_y = int(sh * 0.20)
        return sx + pad_x, sy + pad_y, min(img_w, sx + sw - pad_x), min(img_h, sy + sh - pad_y)

    visible_indices = list(range(num_visible))

    # --- Step 4.2.2: Hintergrund-ROI (Background Noise) berechnen ---
    last_rel_idx = visible_indices[-1]
    prev_rel_idx = visible_indices[-2] if len(visible_indices) > 1 else last_rel_idx

    x_a, _, x_c, _ = get_inset_roi(steps_map[last_rel_idx]) 
    sq_w = x_c - x_a                                        

    _, bg_b, _, _ = get_inset_roi(steps_map[prev_rel_idx]) 
    _, _, _, bg_d = get_inset_roi(steps_map[last_rel_idx]) 

    bg_w = sq_w * 1.5   
    gap_x = sq_w * 1.0  
    
    bg_c = int(max(0, x_a - gap_x))
    bg_a = int(max(0, bg_c - bg_w))
    bg_b = int(bg_b)
    bg_d = int(bg_d)

    # Export Hintergrund-ROI
    save_roi(bg_a, bg_b, bg_c, bg_d, "ROI_Background_Noise", folder)

    # --- Step 4.2.3: Stufen-ROIs berechnen und exportieren ---
    for idx, step_num in enumerate(range(start_step, end_step + 1)):
        sa, sb, sc, sd = get_inset_roi(steps_map[idx])
        save_roi(sa, sb, sc, sd, f"ROI_Step_{step_num}", folder)

    # --- Step 4.2.4: Erstellung des Kontrollbildes (debug_preview.png) ---
    preview = cv2.cvtColor(avg_img, cv2.COLOR_GRAY2BGR)
    
    for idx, step_num in enumerate(range(start_step, end_step + 1)):
        sa, sb, sc, sd = get_inset_roi(steps_map[idx])
        color = (0, 255, 0) if step_num == 7 else (0, 0, 255) if step_num == 1 else (128, 128, 128)
        cv2.rectangle(preview, (int(sa), int(sb)), (int(sc), int(sd)), color, 2)

    cv2.rectangle(preview, (bg_a, bg_b), (bg_c, bg_d), (255, 0, 0), 2)

    debug_path = os.path.join(folder, "..", "debug_preview.png")
    cv2.imwrite(debug_path, preview)
    print(f"Visuelles Kontrollbild gespeichert: {debug_path}")



# SCHRITT 5: MAIN PROGRAMM-STEUERUNG


if __name__ == "__main__":
    try:
        # SCHRITT 1
        folder = get_folder_path()
        print(f"\nVerarbeite Ordner: {folder}")
        
        # SCHRITT 2
        avg_img, count = create_average_image(folder)
        print(f"Erfolgreich {count} Bilder gemittelt.")

        start_step, end_step = get_step_range()

        # SCHRITT 3 & 4
        process_rois(avg_img, folder, start_step, end_step)
        print("\nROIs erfolgreich generiert!")

    except Exception as e:
        print(f"\nFehler aufgetreten: {e}")

    input("\nDrücke Enter zum Beenden...")