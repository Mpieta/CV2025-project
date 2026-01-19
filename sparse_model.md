# SparseModel

## 1. Opis Architektury
Model ten to trójwymiarowa sieć neuronowa (3D CNN) zoptymalizowana pod kątem analizy wideo na urządzeniach o ograniczonych zasobach obliczeniowych. Architektura opiera się na koncepcji **Inverted Residuals** oraz **Linear Bottlenecks**, znanych z modelu MobileNetV2, przeniesionych w wymiar czasoprzestrzenny.

### Kluczowe cechy:
* **Sploty Rozdzielne (3D Depthwise Separable Convolutions):** Redukują koszt obliczeniowy o ok. 8-9 razy w porównaniu do standardowych splotów 3D.
* **ReLU6:** Funkcja aktywacji zapewniająca stabilność przy obliczeniach o niskiej precyzji (np. FP16/INT8).
* **Global Average Pooling (3D):** Sprawia, że model jest odporny na różną długość klipów wideo oraz zmienną rozdzielczość wejściową.



## 2. Struktura Bloku Inverted Residual 3D
Każdy blok składa się z trzech etapów:
1.  **Ekspansja (1x1x1):** Zwiększenie liczby kanałów (zazwyczaj 6-krotne).
2.  **Splot Głębokościowy (3x3x3):** Ekstrakcja cech czasoprzestrzennych (Depthwise).
3.  **Projekcja Liniowa (1x1x1):** Powrót do niższej liczby kanałów bez nieliniowości (Linear Bottleneck).

## 3. Specyfikacja Techniczna (Default)

| Parametr | Wartość | Opis |
| :--- | :--- | :--- |
| **Rozmiar wejściowy** | `[B, 3, 16, 128, 128]` | Batch, RGB, Klatki, Wysokość, Szerokość |
| **Liczba parametrów** | ~2.4 mln | Wersja dla `width_mult=1.0` |
| **Wymiar latentny** | 1280 | Liczba cech przed klasyfikatorem |
| **Aktywacja** | ReLU6 | $f(x) = \min(\max(0, x), 6)$ |


### 4. Szczegółowa Konfiguracja Warstw

Model wykorzystuje specyficzną strukturę warstw zdefiniowaną w `interverted_residual_setting`. Poniższa tabela przedstawia propagacje danych w przód:

| Etap | Typ Warstwy | Kernel | Stride (T, H, W) | Exp (t) | Kanały (c) | Bloki (n) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stem** | Conv3d | 3x3x3 | (1, 2, 2) | - | 32 | 1 |
| **Bneck 1** | InvertedResidual | 3x3x3 | (1, 1, 1) | 1 | 16 | 1 |
| **Bneck 2** | InvertedResidual | 3x3x3 | (2, 2, 2) | 6 | 24 | 2 |
| **Bneck 3** | InvertedResidual | 3x3x3 | (2, 2, 2) | 6 | 32 | 3 |
| **Bneck 4** | InvertedResidual | 3x3x3 | (2, 2, 2) | 6 | 64 | 4 |
| **Bneck 5** | InvertedResidual | 3x3x3 | (1, 1, 1) | 6 | 96 | 3 |
| **Bneck 6** | InvertedResidual | 3x3x3 | (2, 2, 2) | 6 | 160 | 3 |
| **Bneck 7** | InvertedResidual | 3x3x3 | (1, 1, 1) | 6 | 320 | 1 |
| **Final** | Conv3d | 1x1x1 | (1, 1, 1) | - | 1280 | 1 |
| **Pooling** | AdaptiveAvg | Global | - | - | 1280 | 1 |
| **Head** | Linear | - | - | - | 17 | 1 |


## 5. Warianty Architektury (Tryby Pracy)

Kod umożliwia wybór trybu działania sieci poprzez parametr `mode`. Pozwala to na balansowanie między szybkością inferencji a zdolnością do rozpoznawania szybkich ruchów.

| Cecha | Mode: `standard` (Domyślny) | Mode: `high_temporal` |
| :--- | :--- | :--- |
| **Strategia Stride'u** | Agresywna redukcja czasu i przestrzeni `(2, 2, 2)` | Zachowanie czasu, redukcja przestrzeni `(1, 2, 2)` |
| **Rozdzielczość Czasowa** | Niska (Features: 1 klatka) | Wysoka (Features: 16 klatek*) |
| **Zużycie Pamięci (VRAM)** | Bardzo Niskie | Średnie (ok. 2-3x wyższe mapy cech) |
| **Zastosowanie** | Ćwiczenia statyczne, wolne ruchy, urządzenia mobile | Szybkie ruchy (boks, skoki), analiza dynamiki |
| **Skip Connections** | Tylko w blokach bez redukcji | Tylko w blokach bez redukcji (identyczna topologia) |


