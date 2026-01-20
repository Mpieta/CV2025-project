# Grad-CAM / HiResCAM

Grad-CAM (Gradient-weighted Class Activation Mapping) i HiResCAM to techniki wizualizacji, które pokazują, które regiony wejścia są istotne dla decyzji klasyfikatora. Generują heatmapę nakładaną na oryginalny obraz/wideo. Grad-CAM używa Global Average Pooling do agregacji gradientów, natomiast HiResCAM mnoży gradienty z aktywacjami element-wise, co pozwala zachować wyższą rozdzielczość przestrzenną.
Do wizualizacji używamy ostatniego bloku konwolucyjnego przed klasyfikatorem, gdyż posiada najwyższy poziom abstrakcji semantycznej.

## Zasada działania

1. Forward pass - zapisanie aktywacji z target layer
2. Backward pass - obliczenie gradientów względem klasy docelowej
3. Ważenie map aktywacji:
   - **Grad-CAM**: Global Average Pooling gradientów, następnie ważona suma aktywacji
   - **HiResCAM**: Element-wise multiplication gradientów i aktywacji (wyższa rozdzielczość)
4. Interpolacja do rozmiaru wejścia
5. Normalizacja i nałożenie na obraz


## Wizualizacja
![img.png](img.png)

