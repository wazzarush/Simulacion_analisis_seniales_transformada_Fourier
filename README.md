# Simulación y análisis de señales con la Transformada de Fourier

Proyecto de la **Actividad formativa 2** en Python. Muestra cómo se representan algunas señales en el dominio del tiempo y en el dominio de la frecuencia, y verifica numéricamente las propiedades principales de la Transformada de Fourier.

## ¿Qué hace el código?

El archivo [`fourier_senales.py`](fourier_senales.py) está organizado en secciones comentadas:

1. **Definición de la señal en el dominio del tiempo:** pulso rectangular, escalón unitario y señal senoidal de 5 Hz.
2. **Cálculo de la Transformada de Fourier** con `np.fft.fft()` (con `fftshift` y escalamiento por `dt` para aproximar la integral continua).
3. **Visualización:** gráficas de la señal, la magnitud |X(f)| y la fase ∠X(f).
4. **Análisis de propiedades:**
   - Linealidad: `F{a·x1 + b·x2} = a·F{x1} + b·F{x2}`
   - Desplazamiento en el tiempo: `F{x(t − t0)} = X(f)·e^(−j2πf·t0)`
   - Escalamiento: `F{x(a·t)} = (1/|a|)·X(f/a)`
5. **Aplicación real (simulada):** eliminación de la interferencia de 60 Hz de la red eléctrica en un electrocardiograma (ECG).

## Requisitos e instalación

- Python 3.9 o superior
- NumPy y Matplotlib

```bash
pip install -r requirements.txt
```

```powershell
python -m pip install -r requirements.txt
```
## Ejecución

```bash
python fourier_senales.py
```

El programa imprime en pantalla los errores numéricos de cada verificación y guarda las gráficas en la carpeta [`figuras/`](figuras/).

## Resultados principales

| Verificación | Resultado |
|---|---|
| Pulso rectangular vs. sinc teórica | error máximo ≈ 6×10⁻⁴ |
| Senoidal de 5 Hz | picos en ±5 Hz, fase de −π/2 en +5 Hz |
| Linealidad | error ≈ 4×10⁻¹⁶ |
| Desplazamiento en el tiempo | la magnitud no cambia; la fase cambia −2πf·t0 (error ≈ 10⁻¹⁶) |
| Escalamiento (a = 2) | error relativo ≈ 5×10⁻³ |
| Filtrado de 60 Hz en ECG | el error cuadrático medio baja de 0.357 a 0.050 (−86 %) |

## Gráficas generadas

| Archivo | Contenido |
|---|---|
| `fig1_senales_tiempo.png` | Señales en el dominio del tiempo |
| `fig2a_espectro_pulso.png`, `fig2b_espectro_escalon.png`, `fig2c_espectro_senoidal.png` | Magnitud y fase de cada señal |
| `fig3_linealidad.png` | Verificación de la linealidad |
| `fig4_desplazamiento.png` | Verificación del desplazamiento en el tiempo |
| `fig5_escalamiento.png` | Verificación del escalamiento |
| `fig6_aplicacion_ecg.png` | Filtrado de 60 Hz en un ECG simulado |

## Estructura del repositorio

```
fourier-senales/
├── fourier_senales.py
├── requirements.txt
├── README.md
└── figuras/
```

## Notas

- Todas las señales son simuladas; el ECG es una señal sintética construida con ondas gaussianas.
- La FFT supone que la señal es periódica dentro de la ventana de observación. Por eso el escalón se comporta como una onda cuadrada de periodo 2 s, y su espectro tiene líneas en las frecuencias que son múltiplos impares de 0.5 Hz.

## Autor

- Abigail Ramos
