// ==========================================
// FLUORO-QC - Stack Version (ImageJ / Fiji Macro)
// Autor der Anpassung: Hakim Kayed
// Kontext: Masterarbeit zur Untersuchung der Bildqualität von Angiografiebildern
// Zitat: Erstellt, übersetzt und optimiert unter Nutzung von Google Gemini 3.1 Pro 
//        zur Übersetzung von Konzepten/Ideen in Makro-Code sowie zur Code-Optimierung.
//
// Wissenschaftliche Grundlage:
// Vano E, Ubeda C, Geiger B, Martinez LC, Balter S.
// "Influence of image metrics when assessing image quality from a test object in cardiac X-ray systems"
// J Digit Imaging. 2011 Apr;24(2):331-8. doi: 10.1007/s10278-009-9268-7.
// 
// Zweck: Automatische Qualitätskontroll-Analyse (QC) von Durchleuchtungs-Stacks 
//        mittels ROI-Verarbeitung (ROI_Background_Noise & ROI_Step_X) zur 
//        Bestimmung von SNR, BG_STD sowie Grauwerterfassung und CNR für JEDE Stufe
//        inkl. Standardabweichung (Fehler) und ROI-Flächenanzeige.
// ==========================================


// SCHRITT 0: RESET & AUFRÄUMEN VOR JEDEM RUN
// -------------------------------------------------
if (isOpen("Results")) {
    selectWindow("Results");
    run("Close");
}
if (isOpen("Log")) {
    selectWindow("Log");
    run("Close");
}


// SCHRITT 1: BILD-PREPROCESSING & SETUP
// -------------------------------------------------
if (nImages == 0) {
    exit("Fehler: Kein Bild geladen! Das Bild muss vor der Makroausführung geöffnet sein.");
}

if (bitDepth == 24)
    run("RGB to Grayscale");

run("8-bit");
run("Set Measurements...", "area mean standard min integrated display redirect=None decimal=5");


// SCHRITT 2: STACK- & ROI-VALIDIERUNG & INDIZIERUNG
// -------------------------------------------------
totalSlices = nSlices;
if (totalSlices < 1) totalSlices = 1;

numRois = roiManager("count");
if (numRois < 2) {
    exit("Fehler: Bitte lade mindestens die 'ROI_Background_Noise' und mindestens eine 'ROI_Step_X' in den ROI Manager!");
}

bgIndex = -1;
stepIndices = newArray(0);
stepNumbers = newArray(0);

for (r = 0; r < numRois; r++) {
    roiManager("select", r);
    roiName = getInfo("selection.name");
    
    if (indexOf(roiName, "Background") >= 0 || indexOf(roiName, "BG") >= 0) {
        bgIndex = r;
    } else if (indexOf(roiName, "Step_") >= 0) {
        stepIndices = Array.concat(stepIndices, r);
        parts = split(roiName, "_");
        numStr = parts[parts.length - 1]; 
        stepNumbers = Array.concat(stepNumbers, parseInt(numStr));
    }
}

run("Select None");

if (bgIndex == -1) {
    exit("Fehler: Keine Hintergrund-ROI ('ROI_Background_Noise') im ROI Manager gefunden!");
}

numSteps = stepIndices.length;
if (numSteps == 0) {
    exit("Fehler: Keine Stufen-ROIs ('ROI_Step_X') im ROI Manager gefunden!");
}


// SCHRITT 3: FLÄCHEN (AREA) DER ROIS ERFASSEN
// -------------------------------------------------
roiManager("select", bgIndex);
getStatistics(bgArea);

stepAreas = newArray(numSteps);
for (s = 0; s < numSteps; s++) {
    roiManager("select", stepIndices[s]);
    getStatistics(sArea);
    stepAreas[s] = sArea;
}
run("Select None");


// SCHRITT 4: DEFINITION DES ANALYSE-BEREICHS (SLICES)
// -------------------------------------------------
startSlice = 10; 
endSlice   = 75; 

if (endSlice > totalSlices)
    endSlice = totalSlices;

countSlices = (endSlice - startSlice) + 1;
if (countSlices <= 0) {
    exit("Fehler: Ungültiger Slice-Bereich!");
}


// SCHRITT 5: INITIALISIERUNG DER DATEN-SPEICHER (FÜR SD-BERECHNUNG)
// -------------------------------------------------
bgMeanArray = newArray(countSlices);
bgStdArray  = newArray(countSlices);
snrArray    = newArray(countSlices);

stepMeanValues = newArray(numSteps * countSlices);
stepCnrValues  = newArray(numSteps * countSlices);


// SCHRITT 6: SCHLEIFE: SLICE-BY-SLICE ANALYSE
// -------------------------------------------------
row = 0;         
sliceIdx = 0;

for (i = startSlice; i <= endSlice; i++) {
    
    setSlice(i); 

    // --- 1. HINTERGRUND-ROI (BG) ---
    run("Select None");
    roiManager("select", bgIndex);
    getStatistics(area, mean, min, max, std);
    
    BG_MEAN = mean;
    BG_STD  = std;
    SNR     = BG_MEAN / BG_STD;

    bgMeanArray[sliceIdx] = BG_MEAN;
    bgStdArray[sliceIdx]  = BG_STD;
    snrArray[sliceIdx]    = SNR;

    setResult("Slice",   row, i);
    setResult("BG_MEAN", row, BG_MEAN);
    setResult("BG_STD",  row, BG_STD);
    setResult("SNR",     row, SNR);

    // --- 2. STUFEN-ROIS (DYNAMISCH FÜR JEDE STUFE) ---
    for (s = 0; s < numSteps; s++) {
        rIdx = stepIndices[s];
        sNum = stepNumbers[s];

        run("Select None");
        roiManager("select", rIdx);
        getStatistics(area, mean, min, max, std);
        
        stepMean = mean;
        stepCNR = abs(stepMean - BG_MEAN) / BG_STD;

        stepMeanValues[s * countSlices + sliceIdx] = stepMean;
        stepCnrValues[s * countSlices + sliceIdx]  = stepCNR;

        setResult("MEAN_Step_" + sNum, row, stepMean);
        setResult("CNR_Step_" + sNum,  row, stepCNR);
    }

    row++;         
    sliceIdx++; 
}


// SCHRITT 7: MITTELWERTE & STANDARDABWEICHUNGEN (FEHLER) BERECHNEN
// -------------------------------------------------
function getMean(arr) {
    sum = 0;
    for (k = 0; k < arr.length; k++) sum += arr[k];
    return sum / arr.length;
}

function getSD(arr, meanVal) {
    if (arr.length <= 1) return 0;
    sumSqDiff = 0;
    for (k = 0; k < arr.length; k++) {
        diff = arr[k] - meanVal;
        sumSqDiff += diff * diff;
    }
    return sqrt(sumSqDiff / (arr.length - 1));
}

meanBG_MEAN = getMean(bgMeanArray);
sdBG_MEAN   = getSD(bgMeanArray, meanBG_MEAN);

meanBG_STD  = getMean(bgStdArray);
sdBG_STD    = getSD(bgStdArray, meanBG_STD);

meanSNR     = getMean(snrArray);
sdSNR       = getSD(snrArray, meanSNR);

meanStepMeans = newArray(numSteps);
sdStepMeans   = newArray(numSteps);
meanStepCNRs  = newArray(numSteps);
sdStepCNRs    = newArray(numSteps);

for (s = 0; s < numSteps; s++) {
    tmpMeans = newArray(countSlices);
    tmpCNRs  = newArray(countSlices);
    for (k = 0; k < countSlices; k++) {
        tmpMeans[k] = stepMeanValues[s * countSlices + k];
        tmpCNRs[k]  = stepCnrValues[s * countSlices + k];
    }
    
    meanStepMeans[s] = getMean(tmpMeans);
    sdStepMeans[s]   = getSD(tmpMeans, meanStepMeans[s]);
    
    meanStepCNRs[s]  = getMean(tmpCNRs);
    sdStepCNRs[s]    = getSD(tmpCNRs, meanStepCNRs[s]);
}


// SCHRITT 8: FORMATIERTE LOG-AUSGABE & STRUCTURED SUMMARY (DEUTSCHE DEZIMALZEICHEN)
// -------------------------------------------------
function formatVal(val) {
    return replace(d2s(val, 5), "\\.", ",");
}

print("================================================================================");
print("METHODEN-REFERENZ & DOKUMENTATION");
print("================================================================================");
print("Wissenschaftliche Grundlage:");
print("  Vano E, Ubeda C, Geiger B, Martinez LC, Balter S.");
print("  Influence of image metrics when assessing image quality from a test object in cardiac X-ray systems");
print("  J Digit Imaging. 2011 Apr;24(2):331-8. doi: 10.1007/s10278-009-9268-7.");
print(" ");
print("Script-Anpassung & Implementierung:");
print("  Autor:\t\tHakim Kayed");
print("  Kontext:\t\tMasterarbeit zur Untersuchung der Bildqualität von Angiografiebildern");
print("  Hinweis / Zitat:\tErstellt, übersetzt und optimiert unter Nutzung von Google Gemini 3.1 Pro"); 
print("                  \tzur Übersetzung von Konzepten/Ideen in Python-Code sowie zur Code-Optimierung.");
print("================================================================================");
print(" ");
print("=================================================");
print("                  Ergebnisse:");
print("=================================================");

print("Anzahl analysierter Slices\t=\t" + countSlices);
print("Hintergrund ROI-Fläche (Area)\t=\t" + formatVal(bgArea));
print("Mittelwert BG_MEAN (Grauwert)\t=\t" + formatVal(meanBG_MEAN) + " +/- " + formatVal(sdBG_MEAN));
print("Mittelwert BG_STD\t\t=\t" + formatVal(meanBG_STD) + " +/- " + formatVal(sdBG_STD));
print("Mittelwert SNR\t\t\t=\t" + formatVal(meanSNR) + " +/- " + formatVal(sdSNR));
print("-------------------------------------------------");

print("STUFEN-GRAUWERTE (MEAN +/- SD & AREA):");
for (s = 0; s < numSteps; s++) {
    sNum = stepNumbers[s];
    print("Step " + sNum + " [Fläche: " + formatVal(stepAreas[s]) + "]\t-\tGrauwert = " + formatVal(meanStepMeans[s]) + " +/- " + formatVal(sdStepMeans[s]));
}

print("-------------------------------------------------");

print("STUFEN-CNR (MEAN +/- SD):");
for (s = 0; s < numSteps; s++) {
    sNum = stepNumbers[s];
    print("Step " + sNum + " - CNR\t\t\t=\t" + formatVal(meanStepCNRs[s]) + " +/- " + formatVal(sdStepCNRs[s]));
}
print("=================================================");
print(" ");

updateResults();


// SCHRITT 9: ORDNER ERSTELLEN, ERGEBNISSE SPEICHERN & SAUBER SCHLIESSEN
// -------------------------------------------------
imageDir  = getDirectory("image");
imageName = getTitle();

if (imageDir == "") {
    imageDir = getDirectory("home");
}

saveDirName = "Log-" + imageName;
outputDir   = imageDir + saveDirName + File.separator;

File.makeDirectory(outputDir);

// 1. Log-Fenster als Textdatei speichern
logPath = outputDir + "QC_Log.txt";
selectWindow("Log");
saveAs("Text", logPath);

// 2. Slices-Einzelergebnisse als CSV-Datei speichern
csvPath = outputDir + "QC_Results_Slices.csv";
saveAs("Results", csvPath);

// 3. Zusammenfassung als Excel-kompatibles CSV speichern
summaryCsvPath = outputDir + "QC_Summary.csv";
f = File.open(summaryCsvPath);

File.append("Metrik;Wert;Fehler_SD;Flaeche_Area", summaryCsvPath);
File.append("Anzahl analysierter Slices;" + countSlices + ";-;-", summaryCsvPath);
File.append("Background Noise;" + formatVal(meanBG_MEAN) + ";" + formatVal(sdBG_MEAN) + ";" + formatVal(bgArea), summaryCsvPath);
File.append("Mittelwert BG_STD;" + formatVal(meanBG_STD) + ";" + formatVal(sdBG_STD) + ";-", summaryCsvPath);
File.append("Mittelwert SNR;" + formatVal(meanSNR) + ";" + formatVal(sdSNR) + ";-", summaryCsvPath);

for (s = 0; s < numSteps; s++) {
    sNum = stepNumbers[s];
    File.append("Step " + sNum + " - Grauwert (Mean);" + formatVal(meanStepMeans[s]) + ";" + formatVal(sdStepMeans[s]) + ";" + formatVal(stepAreas[s]), summaryCsvPath);
}

for (s = 0; s < numSteps; s++) {
    sNum = stepNumbers[s];
    File.append("Step " + sNum + " - CNR;" + formatVal(meanStepCNRs[s]) + ";" + formatVal(sdStepCNRs[s]) + ";-", summaryCsvPath);
}

if (isOpen("Log")) {
    selectWindow("Log");
    run("Close");
}
if (isOpen("Results")) {
    selectWindow("Results");
    run("Close");
}