/*
Autor 1: Hakim Kayed

Verwendungszweck: Masterarbeit zur Untersuchung der Bildqualität von Angiografiebildern
Hinweis / Zitat: Erstellt, übersetzt und optimiert unter Nutzung von Google Gemini 3.1 Pro 
                 zur Übersetzung von Konzepten/Ideen in Makro-Code sowie zur Code-Optimierung.
Zweck: Auslesen des Linienprofils der aktuell ausgewählten ROI und Übertragung in die ResultsTable.
*/


// SCHRITT 1: BILDVORVERARBEITUNG
// Falls ein RGB-Bild vorliegt, in Graustufen umwandeln und auf 8-Bit setzen
if (bitDepth() == 24) {
    run("RGB to Grayscale");
}
run("8-bit");


// SCHRITT 2: PROFIL DER AKTUELLEN ROI AUSLESEN
// Profildaten der im RoiManager ausgewählten ROI erfassen und validieren
prof = getProfile();
nPoints = lengthOf(prof);

if (nPoints < 1) {
    exit("Fehler: Die ausgewählte ROI liefert kein Profil.");
}


// SCHRITT 3: ERGEBNISTABELLE BEFÜLLEN UND AKTUALISIEREN
// Tabelleninhalt zurücksetzen und X/Y-Profilwerte eintragen
run("Clear Results");

for (j = 0; j < nPoints; j++) {
    setResult("X", j, j);
    setResult("Y", j, prof[j]); // Schreibt den Grauwert in Spalte 'Y'
}

updateResults();