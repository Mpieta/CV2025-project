# Raport z Implementacji Augmentacji Wideo 
# Autor: Radosław Mocarski

## 1. Cel Zmian

Celem modyfikacji było wprowadzenie zaawansowanej augmentacji danych wideo "w locie" (on-the-fly), aby zwiększyć zdolność generalizacji modelu i zapobiec overfittingowi. Kluczowym wymaganiem było zachowanie **spójności czasowej** (temporal consistency) – transformacje geometryczne (np. obrót) muszą być identyczne dla każdej klatki w obrębie jednego klipu wideo.

## 2. Podsumowanie Zmian

### Co zostało dodane:

* **Biblioteka `torchvision.transforms.v2`**: Zastąpiono standardowe transformacje nowym API, które obsługuje tensory wideo i automatycznie aplikuje tę samą losową transformację do całej sekwencji klatek.
* **Tryb Debugowania (`--debug`)**: Dodano flagę w `Trainer.py` oraz logikę w `Dataset.py`, która zapisuje próbki wideo "przed" i "po" augmentacji na dysku (folder `debug_output`). Pozwala to na wizualną weryfikację poprawności danych.
* **Zapis Wideo (OpenCV)**: Zaimplementowano funkcję `_save_debug_video` w `Dataset.py`, która konwertuje tensory z powrotem na pliki `.mp4` w celu podglądu.

### Co zostało zmienione:

* **Rozdzielczość Wideo**: Zmniejszono docelową rozdzielczość wejściową sieci do **300x400**, aby ujednolicić dane wyjściowe i ograniczyć zapotrzebowanie na pamięć RAM i VRAM.
* **Logika `__getitem__`**: Przebudowano sposób ładowania danych. Transformacje nie są już wykonywane w pętli na pojedynczych obrazkach. Zamiast tego, klatki są najpierw łączone w jeden tensor [T, C, H, W], a następnie transformowane całościowo.
* **Zarządzanie Pamięcią**: W trybie debugowania wyłączono cache'owanie RAM, aby zawsze generować nowe warianty augmentacji.
* **Obsługa Argumentów**: `Trainer.py` obsługuje teraz argumenty z linii komend (`argparse`).

### Co zostało usunięte:

* Stara metoda transformacji aplikowana wewnątrz pętli `for` podczas czytania klatek (powodowała "migotanie" obrazu, ponieważ każda klatka była losowana osobno).

---

## 3. Szczegóły Augmentacji

Zastosowano potok transformacji w `torchvision.transforms.v2`, który obejmuje:

1. **Resize**: Skalowanie obrazu z zachowaniem proporcji (krótszy bok dopasowany do wymiaru bazowego).
2. **RandomResizedCrop (do 300x400)**: Losowe wycięcie fragmentu obrazu i przeskalowanie do docelowej rozdzielczości. Symuluje to zmianę odległości kamery (zoom) i drobne przesunięcia kadru.
3. **RandomHorizontalFlip (p=0.5)**: Losowe odbicie lustrzane (kluczowe w fitnessie – uczy model niezależności od strony ciała).
4. **RandomRotation (+/- 10 stopni)**: Symulacja nagrywania "z ręki" lub krzywo ustawionego statywu.
5. **Normalization**: Standaryzacja wartości pikseli (ImageNet stats) dla szybszej zbieżności treningu.

---

## 4. Instrukcja Uruchomienia

Skrypt `Trainer.py` przyjmuje teraz parametry sterujące procesem.

### A. Tryb Weryfikacji (Debug)

Służy do sprawdzenia, czy augmentacja działa poprawnie i czy dane "mają sens" wizualnie.

* Trenuje tylko na 5% danych (szybki sanity check).
* Zapisuje pliki wideo (oryginał vs augmentacja) w folderze `debug_output`.
* Działa na 1 wątku (dla stabilności zapisu plików na Windows).

```bash
python Trainer.py --debug True

```

### B. Pełny Trening

Uruchamia właściwy proces uczenia na pełnym zbiorze danych.

* Wykorzystuje 100% batchy treningowych.
* Używa wielowątkowości (`num_workers=4` w `Dataset.py`) dla maksymalnej wydajności.
* Nie zapisuje plików wideo na dysku.

```bash
python Trainer.py --debug False --model mobilenet_v2_3d --epochs 10

```

*(Opcjonalnie można wybrać model `pretrained_resnet` zamiast `mobilenet_v2_3d`)*.

---

## 5. Konfiguracja Techniczna

| Parametr | Wartość | Uwagi |
| --- | --- | --- |
| **Rozdzielczość** | **300x400** (WxH) | Ustandaryzowana jakość detali |
| **Długość klipu** | 16 klatek | Standard dla modeli 3D CNN |
| **Batch Size** | 16 | Dostosuj w dół, jeśli wystąpi `CUDA OOM` |
| **Framework** | PyTorch + Torchvision v2 | Wymaga nowszej wersji Torchvision |
| **Workerzy** | 0 (Debug) / 4 (Train) | Zdefiniowane w `Dataset.py` |

---

## 6. Optymalizacja Wydajności (CPU Workers)

Liczba wątków procesora (`num_workers`) odpowiedzialnych za ładowanie i augmentację danych jest kluczowa dla szybkości treningu.

*   **Lokalizacja zmiany**: Plik `Dataset.py`, klasa `ExerciseVideoDataModule`, metody `train_dataloader` i `val_dataloader`.
*   **Zalecenia**:
    *   **Windows**: Zalecana ostrożność. Wartość `0` jest najbezpieczniejsza (główny wątek). Wartości `4-8` mogą przyspieszyć trening, ale mogą powodować błędy przy użyciu OpenCV (`cv2`).
    *   **Linux**: Zalecana wysoka wartość (np. liczba rdzeni CPU, typowo `8` lub `16`).
    *   **Sprzęt**: Zawsze uzależniaj tą liczbę od twojego sprzętu na którym pracujesz. Nie przekraczaj liczby rdzeni na wykorzystywanym CPU, a najlepiej zostaw co najmniej jeden na obsługę innych zadań.
*   **Obecna logika**:
    Kod automatycznie ustawia `num_workers=0` w trybie debugowania (dla stabilności), oraz `num_workers=4` w trybie pełnego treningu. Aby to zmienić ręcznie, edytuj argument w wywołaniu `DataLoader`:

    ```python
    # Przykład w Dataset.py
    return DataLoader(..., num_workers=8, ...) # Zwiększenie do 8 wątków
    ```