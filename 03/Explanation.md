# K-Means Clustering - Dokumentation für Anfänger

Diese Dokumentation erklärt wichtige Funktionen und Codezeilen aus dem K-Means Clustering Programm.

---

## 1. Die `kmeans()` Funktion

### Was ist K-Means überhaupt?

Stell dir vor, du hast einen Haufen bunter Murmeln auf dem Boden verstreut. Einige sind rot, andere blau, grün, gelb und orange. Deine Aufgabe ist es, diese Murmeln in 5 verschiedene Schachteln zu sortieren, wobei ähnliche Murmeln zusammenkommen sollen. Aber es gibt einen Haken: Die Murmeln haben keine Etiketten drauf! Du musst also selbst herausfinden, welche Murmeln zusammengehören.

Genau das macht der K-Means Algorithmus, nur mit Datenpunkten statt Murmeln!

### Wie funktioniert die Funktion?

```python
def kmeans(features, k, num_iters=100):
```

**Was bedeuten die Eingabewerte (Parameter)?**

- **`features`**: Das ist deine Sammlung von Datenpunkten (die "Murmeln"). Jeder Punkt hat bestimmte Eigenschaften, z.B. Position, Farbe, Größe
- **`k`**: Die Anzahl der Gruppen (Cluster), in die du deine Daten aufteilen möchtest. Bei unserem Murmel-Beispiel wären das 5 Schachteln
- **`num_iters=100`**: Wie oft der Algorithmus maximal versuchen soll, die Gruppen zu verbessern. Standard sind 100 Versuche

### Der Algorithmus Schritt-für-Schritt:

**Schritt 1: Zufällige Startpunkte wählen**
```python
idxs = np.random.choice(N, size=k, replace=False)
centers = features[idxs]
```
Stell dir vor, du schließt die Augen und wirfst zufällig 5 leere Schachteln auf den Boden. Das sind deine Start-"Zentren". Von hier aus beginnt die Sortierung.

**Schritt 2: Distanzen berechnen**
```python
distances = np.zeros((N, k))
for i in range(k):
    distances[:, i] = np.linalg.norm(features - centers[i], axis=1)
```
Jetzt misst du für jede einzelne Murmel: "Wie weit ist diese Murmel von jeder Schachtel entfernt?" Das ist wie mit einem Lineal die Entfernung zu messen.

**Schritt 3: Zuordnung zu nächstem Zentrum**
```python
new_assignments = np.argmin(distances, axis=1)
```
Jede Murmel kommt in die Schachtel, die am nächsten ist. Wenn eine rote Murmel näher an Schachtel 1 ist als an allen anderen, kommt sie in Schachtel 1.

**Schritt 4: Prüfen ob fertig**
```python
if np.all(assignments == new_assignments):
    break
```
"Hat sich etwas verändert? Sind die Murmeln jetzt in anderen Schachteln als vorher?" Wenn nicht, sind wir fertig! Wenn doch, machen wir weiter.

**Schritt 5: Neue Zentren berechnen**
```python
for i in range(k):
    cluster_points = features[assignments == i]
    if len(cluster_points) > 0:
        centers[i] = np.mean(cluster_points, axis=0)
```
Jetzt kommt der clevere Teil: Wir verschieben jede Schachtel genau in die Mitte aller Murmeln, die darin liegen! Das ist, als würdest du den "Schwerpunkt" der Gruppe finden. Dann fangen wir wieder bei Schritt 2 an.

**Das Ergebnis:**
Die Funktion gibt zurück, welche Murmel in welcher Schachtel gelandet ist (die "assignments").

---

## 2. Die `compute_accuracy()` Funktion

### Was ist "Accuracy" (Genauigkeit)?

Stell dir vor, du hast einen Test mit 10 Fragen geschrieben. Du hast 8 richtig und 2 falsch. Deine Genauigkeit ist 8/10 = 0.8 oder 80%.

Die `compute_accuracy()` Funktion macht genau das für unseren K-Means Algorithmus!

### So funktioniert's:

```python
def compute_accuracy(y_true, y_pred):
```

**Die Eingabewerte:**
- **`y_true`**: Die richtigen Antworten (wie der Lösungsschlüssel beim Test)
- **`y_pred`**: Deine Antworten/Vorhersagen (was der Algorithmus gedacht hat)

### Der Code erklärt:

```python
true = 0  # Zähler für richtige Antworten

for i in range(len(y_true)):
    if y_true[i] == y_pred[i]:
        true+=1
```

**Was passiert hier?**
1. Wir erstellen einen Zähler, der bei 0 startet
2. Wir gehen durch jede einzelne Vorhersage
3. Wenn die Vorhersage richtig ist (stimmt mit der Wahrheit überein), zählen wir +1

```python
accuracy = true / len(y_true)
```

**Das Endergebnis:**
Wir teilen die Anzahl der richtigen Antworten durch die Gesamtzahl aller Antworten. Das gibt uns einen Wert zwischen 0 und 1:
- **1.0 = 100%** → Perfekt! Alles richtig!
- **0.5 = 50%** → Die Hälfte richtig
- **0.0 = 0%** → Alles falsch

---

## 3. Die `compute_confusion_matrix()` Funktion

### Was ist eine Confusion Matrix (Verwechslungsmatrix)?

Stell dir vor, du hast einen Freund, der immer Hunde und Katzen verwechselt. Eine Confusion Matrix zeigt dir genau, WAS er WIE OFT verwechselt!

**Beispiel:**
- Du zeigst ihm 10 Katzenbilder → Er erkennt 7 als Katzen (gut!), aber 3 nennt er Hunde (verwechselt!)
- Du zeigst ihm 10 Hundebilder → Er erkennt 9 als Hunde (super!), aber 1 nennt er Katze (hoppla!)

Die Matrix sieht dann so aus:

```
              Vorhergesagt:
              Katze  Hund
Wirklich:
Katze           7      3
Hund            1      9
```

### So funktioniert der Code:

```python
def compute_confusion_matrix(y_true, y_pred):
```

**Schritt 1: Größe bestimmen**
```python
size = len(np.unique(y_true))
```
Wie viele verschiedene Kategorien gibt es? (Im Beispiel: Katzen und Hunde = 2)

**Schritt 2: Leere Matrix erstellen**
```python
confusion_matrix = np.zeros((size, size))
```
Wir erstellen eine Tabelle voller Nullen. Bei 5 Klassen ist das eine 5×5 Tabelle.

**Schritt 3: Matrix füllen**
```python
for i in range(len(y_true)):
    confusion_matrix[y_true[i], y_pred[i]]+= 1
```

**Was passiert hier?**
Für jedes Objekt:
1. Schaue, was es WIRKLICH ist (Zeile)
2. Schaue, was vorhergesagt wurde (Spalte)
3. Mache einen Strich in dieser Zelle (+1)

**Beispiel:**
- Objekt ist wirklich Klasse 0, wurde vorhergesagt als Klasse 2
- → confusion_matrix[0, 2] wird um 1 erhöht

**Das Endergebnis:**
Eine Tabelle, die zeigt, welche Klassen oft verwechselt werden. Die Diagonale (oben links nach unten rechts) zeigt die richtigen Vorhersagen!

---

## 4. Die Zeile: `X_flat = X.reshape(X.shape[0], -1)`

### Was ist ein Bild im Computer?

Ein digitales Bild ist wie ein Raster aus bunten Quadraten (Pixeln). Unser CIFAR-10 Bild hat:
- **32 Pixel** breit
- **32 Pixel** hoch
- **3 Farbkanäle** (Rot, Grün, Blau)

Das ist wie ein Würfel mit den Maßen 32 × 32 × 3.

### Was bedeutet "reshape" (Umformen)?

Stell dir vor, du hast einen Zauberwürfel (3D-Objekt). Mit `reshape` verwandelst du ihn in eine lange Schlange (1D-Linie), ohne dabei Teile zu verlieren oder hinzuzufügen!

### Der Code im Detail:

```python
X_flat = X.reshape(X.shape[0], -1)  # (2000, 32x32x3)
```

**Was bedeutet das?**

- **`X`**: Unsere Bildersammlung
- **`X.shape[0]`**: Die Anzahl der Bilder (2000 Stück)
- **`-1`**: "Computer, rechne selbst aus, wie lang die Zeile sein muss!"
  - 32 × 32 × 3 = 3072 Pixel pro Bild

**Vorher:**
```
Bild 1: [32 × 32 × 3] Würfel
Bild 2: [32 × 32 × 3] Würfel
...
Bild 2000: [32 × 32 × 3] Würfel
```

**Nachher:**
```
Bild 1: [3072 Pixel in einer Reihe]
Bild 2: [3072 Pixel in einer Reihe]
...
Bild 2000: [3072 Pixel in einer Reihe]
```

**Warum machen wir das?**
Viele Algorithmen können nicht mit 3D-Würfeln arbeiten. Sie brauchen flache Listen. Es ist, als würdest du einen Pullover flach hinlegen, bevor du ihn in den Koffer packst!

---

## 5. Die Zeile: `X_std = (X_flat - np.mean(X_flat, axis=0)) / np.std(X_flat, axis=0)`

### Was ist Standardisierung?

Stell dir vor, du vergleichst die Größe von Menschen und das Gewicht von Autos:
- Mensch: 175 cm groß
- Auto: 1500 kg schwer

Das sind völlig unterschiedliche Maßstäbe! Standardisierung bringt alles auf die gleiche Skala.

### Die Zauberformel: z = (x - μ) / σ

**Die Bestandteile:**
- **x**: Ein einzelner Wert (z.B. ein Pixel)
- **μ (mu)**: Der Durchschnitt aller Werte
- **σ (sigma)**: Die Standardabweichung (wie weit die Werte streuen)
- **z**: Der standardisierte Wert

### Schritt-für-Schritt Erklärung:

**Schritt 1: Durchschnitt berechnen**
```python
np.mean(X_flat, axis=0)
```
Für jede Pixel-Position berechnen wir: "Was ist der durchschnittliche Wert über alle 2000 Bilder?"

**Beispiel:**
- Pixel 1 an Position (0,0): Durchschnitt = 120
- Pixel 2 an Position (0,1): Durchschnitt = 85
- usw.

**Schritt 2: Durchschnitt abziehen**
```python
X_flat - np.mean(X_flat, axis=0)
```
Von jedem Pixel-Wert ziehen wir den Durchschnitt ab. Das verschiebt alles so, dass der neue Durchschnitt bei 0 liegt.

**Beispiel:**
- Original-Pixel: 150
- Durchschnitt: 120
- Nach Abzug: 150 - 120 = 30

**Schritt 3: Durch Standardabweichung teilen**
```python
np.std(X_flat, axis=0)
```
Die Standardabweichung sagt: "Wie sehr weichen die Werte vom Durchschnitt ab?"

**Schritt 4: Alles zusammen**
```python
X_std = (X_flat - np.mean(X_flat, axis=0)) / np.std(X_flat, axis=0)
```

**Was haben wir erreicht?**
Jetzt haben alle Pixel:
- **Durchschnitt = 0** (zentriert)
- **Standardabweichung = 1** (normalisiert)

**Ein Vergleich:**
- **Ohne Standardisierung**: Pixel-Werte zwischen 0 und 255
- **Mit Standardisierung**: Werte meist zwischen -3 und +3

### Warum ist das wichtig?

K-Means berechnet Distanzen. Wenn ein Feature (z.B. Helligkeit) Werte von 0-255 hat und ein anderes (z.B. Kontrast) nur 0-1, dominiert die Helligkeit alles! Standardisierung macht alle Features gleich wichtig.

**Analogie:**
Es ist wie bei einem Kuchenrezept: Du musst Gramm in Kilogramm umrechnen und Milliliter in Liter, damit die Proportionen stimmen!

---

## Zusammenfassung

Diese 5 Code-Teile arbeiten zusammen:

1. **`kmeans()`** → Sortiert Daten in Gruppen
2. **`compute_accuracy()`** → Prüft, wie gut die Sortierung war
3. **`compute_confusion_matrix()`** → Zeigt, welche Fehler gemacht wurden
4. **`reshape()`** → Macht aus Bildern flache Listen
5. **Standardisierung** → Bringt alle Werte auf die gleiche Skala

Zusammen helfen sie uns, Bilder automatisch zu kategorisieren – ohne dass wir jedem Bild manuell ein Label geben müssen!