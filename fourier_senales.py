"""
Actividad formativa 2: Simulación y análisis de señales con la Transformada de Fourier
=======================================================================================

Este programa:
  1. Crea señales elementales (pulso rectangular, escalón y senoidal).
  2. Calcula su Transformada de Fourier con np.fft.fft() y grafica magnitud y fase.
  3. Verifica tres propiedades: linealidad, desplazamiento en el tiempo y escalamiento.
  4. Aplica la transformada a un caso real simulado: eliminar la interferencia de 60 Hz
     de una señal de electrocardiograma (ECG).

Uso:
    python fourier_senales.py

Las gráficas se guardan en la carpeta "figuras/".
Requisitos: numpy y matplotlib (ver requirements.txt).
para instalar dependencias: pip install -r requirements.txt 
o
python -m pip install -r requirements.txt

"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # permite guardar figuras sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np

CARPETA = Path(__file__).parent / "figuras"
CARPETA.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# ======================================================================
# 1. DEFINICIÓN DE LA SEÑAL EN EL DOMINIO DEL TIEMPO
# ======================================================================
fs = 1000                       # frecuencia de muestreo (Hz)
dt = 1 / fs                     # periodo de muestreo (s)
N = 2000                        # número de muestras
t = (np.arange(N) - N // 2) * dt          # eje de tiempo: de -1 s a 0.999 s
f = np.fft.fftshift(np.fft.fftfreq(N, dt))  # eje de frecuencia: de -500 a 499.5 Hz


def rectangular(tau, ancho=0.2, centro=0.0):
    """Pulso rectangular de amplitud 1 (los bordes valen 0.5, regla del trapecio)."""
    d = np.abs(tau - centro)
    return np.where(d < ancho / 2 - 1e-9, 1.0,
                    np.where(np.isclose(d, ancho / 2), 0.5, 0.0))


def escalon(tau):
    """Función escalón unitario u(t)."""
    return np.heaviside(tau, 0.5)


def senoidal(tau, amplitud=1.0, frecuencia=5.0):
    """Señal senoidal A*sen(2*pi*f0*t)."""
    return amplitud * np.sin(2 * np.pi * frecuencia * tau)


# ======================================================================
# 2. CÁLCULO DE LA TRANSFORMADA DE FOURIER
# ======================================================================
def transformada(x):
    """
    Aproxima X(f) = integral de x(t)*exp(-j*2*pi*f*t) dt con la FFT.
      - ifftshift: coloca t = 0 en la primera muestra (referencia de fase correcta).
      - np.fft.fft: calcula la DFT.
      - fftshift: ordena las frecuencias de negativas a positivas.
      - * dt: escala la suma para aproximar la integral.
    """
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) * dt


def transformada_en(x, frecuencias):
    """Evalúa la transformada directamente en frecuencias arbitrarias (para verificar el escalamiento)."""
    return np.exp(-2j * np.pi * np.outer(frecuencias, t)) @ x * dt


def fase(X, umbral=1e-6):
    """Fase de X en radianes; se omite donde la magnitud es casi cero (fase sin sentido)."""
    ang = np.angle(X)
    return np.where(np.abs(X) > umbral * np.abs(X).max(), ang, np.nan)


# ======================================================================
# 3. VISUALIZACIÓN DE LOS RESULTADOS
# ======================================================================
def graficar_senal(nombre, archivo, x, rango_f=(-30, 30), teorica=None, discreto=False, umbral_fase=1e-6):
    """Grafica la señal, su magnitud y su fase."""
    X = transformada(x)
    fig, ejes = plt.subplots(1, 3, figsize=(9.5, 3.3))
    ejes[0].plot(t, x, color="#1F3864")
    ejes[0].set(title=f"{nombre} en el tiempo", xlabel="t (s)", ylabel="x(t)", xlim=(-1, 1))
    mask = (f >= rango_f[0]) & (f <= rango_f[1])
    if discreto:
        ejes[1].stem(f[mask], np.abs(X[mask]), basefmt=" ")
    else:
        ejes[1].plot(f[mask], np.abs(X[mask]), color="#2E75B6", label="FFT")
    if teorica is not None:
        ejes[1].plot(f[mask], teorica[mask], "--", color="#C55A11", label="Teórica")
        ejes[1].legend()
    ejes[1].set(title="Magnitud |X(f)|", xlabel="f (Hz)", ylabel="|X(f)|")
    ejes[2].plot(f[mask], fase(X, umbral_fase)[mask], ".", color="#2F8F6B", ms=4)
    ejes[2].set(title="Fase ∠X(f)", xlabel="f (Hz)", ylabel="rad", ylim=(-3.5, 3.5))
    fig.tight_layout()
    fig.savefig(CARPETA / archivo, dpi=150)
    plt.close(fig)
    return X


print("=" * 62)
print("1) SEÑALES ELEMENTALES Y SU TRANSFORMADA DE FOURIER")
print("=" * 62)

x_rect = rectangular(t)                       # pulso de 0.2 s de ancho
x_esc = escalon(t)                            # escalón en t = 0
x_sen = senoidal(t, 1.0, 5.0)                 # senoidal de 5 Hz

# Señales en el dominio del tiempo
fig, ejes = plt.subplots(1, 3, figsize=(9.5, 2.9))
for ax, x, nom in zip(ejes, [x_rect, x_esc, x_sen], ["Pulso rectangular", "Escalón unitario", "Senoidal 5 Hz"]):
    ax.plot(t, x, color="#1F3864")
    ax.set(title=nom, xlabel="t (s)", ylabel="x(t)", xlim=(-1, 1))
fig.tight_layout()
fig.savefig(CARPETA / "fig1_senales_tiempo.png", dpi=150)
plt.close(fig)

# Espectros (magnitud y fase)
teorica_rect = np.abs(0.2 * np.sinc(0.2 * f))      # X(f) = A*T*sinc(f*T)
X_rect = graficar_senal("Pulso rectangular", "fig2a_espectro_pulso.png", x_rect, (-30, 30), teorica_rect)
X_esc = graficar_senal("Escalón", "fig2b_espectro_escalon.png", x_esc, (-10, 10), discreto=True, umbral_fase=2e-2)
X_sen = graficar_senal("Senoidal 5 Hz", "fig2c_espectro_senoidal.png", x_sen, (-15, 15), discreto=True)

err_rect = np.max(np.abs(np.abs(X_rect) - teorica_rect))
k5 = np.argmin(np.abs(f - 5))
print(f"Pulso: X(0) = {X_rect[N // 2].real:.4f} (área del pulso = 0.2); error máx. vs sinc = {err_rect:.2e}")
print(f"Senoidal: |X(5 Hz)| = {np.abs(X_sen[k5]):.4f}, fase = {np.angle(X_sen[k5]):.4f} rad (teoría: -pi/2 = {-np.pi / 2:.4f})")
print(f"Escalón: X(0) = {X_esc[N // 2].real:.4f}")

# ======================================================================
# 4. ANÁLISIS DE PROPIEDADES Y TRANSFORMACIONES
# ======================================================================
print()
print("=" * 62)
print("2) PROPIEDADES DE LA TRANSFORMADA DE FOURIER")
print("=" * 62)

# ---- 4.1 Linealidad: F{a*x1 + b*x2} = a*F{x1} + b*F{x2} ----
a, b = 2.0, -3.0
lado_izq = transformada(a * x_sen + b * x_rect)
lado_der = a * transformada(x_sen) + b * transformada(x_rect)
err_lin = np.max(np.abs(lado_izq - lado_der))
print(f"Linealidad: error máx. |F{{a*x1+b*x2}} - (a*F{{x1}}+b*F{{x2}})| = {err_lin:.2e}")

fig, ejes = plt.subplots(1, 3, figsize=(9.5, 3.3))
ejes[0].plot(t, a * x_sen + b * x_rect, color="#1F3864")
ejes[0].set(title="y(t) = 2·sen(2π5t) − 3·pulso(t)", xlabel="t (s)", xlim=(-1, 1))
m = (f >= -30) & (f <= 30)
ejes[1].plot(f[m], np.abs(lado_izq[m]), color="#2E75B6", lw=3, label="F{a·x1 + b·x2}")
ejes[1].plot(f[m], np.abs(lado_der[m]), "--", color="#C55A11", label="a·F{x1} + b·F{x2}")
ejes[1].set(title="Magnitud: ambos lados coinciden", xlabel="f (Hz)")
ejes[1].legend()
ejes[2].semilogy(f[m], np.abs(lado_izq[m] - lado_der[m]) + 1e-20, color="#2F8F6B")
ejes[2].set(title="Diferencia entre ambos lados", xlabel="f (Hz)", ylabel="error", ylim=(1e-18, 1e-10))
fig.tight_layout()
fig.savefig(CARPETA / "fig3_linealidad.png", dpi=150)
plt.close(fig)

# ---- 4.2 Desplazamiento en el tiempo: F{x(t-t0)} = X(f)*exp(-j*2*pi*f*t0) ----
t0 = 0.3
x_desp = rectangular(t, 0.2, centro=t0)         # mismo pulso, retrasado 0.3 s
X_desp = transformada(x_desp)
X_pred = X_rect * np.exp(-2j * np.pi * f * t0)
err_mag = np.max(np.abs(np.abs(X_desp) - np.abs(X_rect)))
err_desp = np.max(np.abs(X_desp - X_pred))
print(f"Desplazamiento: diferencia máx. de magnitudes = {err_mag:.2e}; error vs X(f)e^(-j2πft0) = {err_desp:.2e}")

valido = (f >= -30) & (f <= 30) & (np.abs(X_rect) > 1e-3)
fig, ejes = plt.subplots(1, 3, figsize=(9.5, 3.3))
ejes[0].plot(t, x_rect, color="#1F3864", label="x(t)")
ejes[0].plot(t, x_desp, color="#C55A11", label="x(t − 0.3)")
ejes[0].set(title="Pulso original y desplazado", xlabel="t (s)", xlim=(-1, 1)); ejes[0].legend()
mm = (f >= -30) & (f <= 30)
ejes[1].plot(f[mm], np.abs(X_rect[mm]), color="#1F3864", lw=3, label="|X(f)|")
ejes[1].plot(f[mm], np.abs(X_desp[mm]), "--", color="#C55A11", label="|X_desp(f)|")
ejes[1].set(title="La magnitud no cambia", xlabel="f (Hz)"); ejes[1].legend()
dif_fase = np.angle(X_desp[valido] / X_rect[valido])
ejes[2].plot(f[valido], dif_fase, ".", color="#2F8F6B", ms=4, label="∠X_desp − ∠X")
ejes[2].plot(f[mm], np.angle(np.exp(-2j * np.pi * f[mm] * t0)), "-", color="#C55A11", lw=1, alpha=0.6, label="−2π f t0 (envuelta)")
ejes[2].set(title="La fase cambia linealmente con f", xlabel="f (Hz)", ylabel="rad"); ejes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(CARPETA / "fig4_desplazamiento.png", dpi=150)
plt.close(fig)

# ---- 4.3 Escalamiento: F{x(a*t)} = (1/|a|) * X(f/a) ----
esc = 2.0
x_esc2 = rectangular(esc * t)                   # x(2t): pulso comprimido a la mitad
X_esc2 = transformada(x_esc2)
X_teo = transformada_en(x_rect, f / esc) / abs(esc)
err_esc = np.max(np.abs(X_esc2 - X_teo)) / np.max(np.abs(X_esc2))
print(f"Escalamiento (a=2): error relativo máx. vs (1/|a|)X(f/a) = {err_esc:.2e}")

fig, ejes = plt.subplots(1, 3, figsize=(9.5, 3.3))
ejes[0].plot(t, x_rect, color="#1F3864", label="x(t): 0.2 s")
ejes[0].plot(t, x_esc2, color="#C55A11", label="x(2t): 0.1 s")
ejes[0].set(title="Compresión en el tiempo", xlabel="t (s)", xlim=(-0.4, 0.4)); ejes[0].legend()
mm = (f >= -50) & (f <= 50)
ejes[1].plot(f[mm], np.abs(X_rect[mm]), color="#1F3864", label="|X(f)|")
ejes[1].plot(f[mm], np.abs(X_esc2[mm]), color="#C55A11", label="|F{x(2t)}|")
ejes[1].set(title="Expansión en frecuencia", xlabel="f (Hz)"); ejes[1].legend()
ejes[2].plot(f[mm], np.abs(X_esc2[mm]), color="#C55A11", lw=3, label="F{x(2t)}")
ejes[2].plot(f[mm], np.abs(X_teo[mm]), "--", color="#2F8F6B", label="(1/2)·X(f/2)")
ejes[2].set(title="Verificación de la propiedad", xlabel="f (Hz)"); ejes[2].legend()
fig.tight_layout()
fig.savefig(CARPETA / "fig5_escalamiento.png", dpi=150)
plt.close(fig)

# Ancho de banda (hasta el primer cero del espectro): 1/ancho
print("Primer cero del espectro: pulso de 0.2 s en 5 Hz; pulso de 0.1 s en 10 Hz")

# ======================================================================
# 5. APLICACIÓN REAL: ELIMINAR LA INTERFERENCIA DE 60 Hz EN UN ECG
# ======================================================================
print()
print("=" * 62)
print("3) APLICACIÓN: FILTRADO DE INTERFERENCIA DE 60 Hz EN UN ECG SIMULADO")
print("=" * 62)

fs_e = 500
t_e = np.arange(0, 10, 1 / fs_e)                 # 10 s de registro
rng = np.random.default_rng(1)                   # semilla fija: resultados reproducibles


def gaussiana(tt, centro, amplitud, ancho):
    return amplitud * np.exp(-((tt - centro) ** 2) / (2 * ancho ** 2))


ecg = np.zeros_like(t_e)
for latido in np.arange(0.4, 10, 60 / 72):       # 72 latidos por minuto
    ecg += gaussiana(t_e, latido - 0.20, 0.15, 0.025)   # onda P
    ecg += gaussiana(t_e, latido - 0.04, -0.12, 0.010)  # onda Q
    ecg += gaussiana(t_e, latido, 1.00, 0.012)          # onda R
    ecg += gaussiana(t_e, latido + 0.04, -0.25, 0.010)  # onda S
    ecg += gaussiana(t_e, latido + 0.28, 0.30, 0.045)   # onda T
ruido = 0.5 * np.sin(2 * np.pi * 60 * t_e) + 0.05 * rng.standard_normal(t_e.size)
ecg_ruidoso = ecg + ruido

E = np.fft.rfft(ecg_ruidoso)                     # espectro (frecuencias >= 0)
fe = np.fft.rfftfreq(t_e.size, 1 / fs_e)
E_filtrado = E.copy()
E_filtrado[(fe >= 58) & (fe <= 62)] = 0          # filtro rechaza-banda en frecuencia
ecg_filtrado = np.fft.irfft(E_filtrado, n=t_e.size)

rmse_antes = np.sqrt(np.mean((ecg_ruidoso - ecg) ** 2))
rmse_despues = np.sqrt(np.mean((ecg_filtrado - ecg) ** 2))
pico = fe[np.argmax(np.abs(E[(fe > 30)])) + np.sum(fe <= 30)]
print(f"Pico de interferencia detectado en: {pico:.1f} Hz")
print(f"Error cuadrático medio antes del filtro: {rmse_antes:.4f}; después: {rmse_despues:.4f}")
print(f"Reducción del error: {100 * (1 - rmse_despues / rmse_antes):.1f} %")

fig, ejes = plt.subplots(1, 3, figsize=(9.5, 3.4))
v = t_e <= 3
ejes[0].plot(t_e[v], ecg_ruidoso[v], color="#7F7F7F", lw=0.8)
ejes[0].set(title="ECG con interferencia de 60 Hz", xlabel="t (s)", ylabel="mV (simulados)")
mfe = fe <= 100
ejes[1].plot(fe[mfe], np.abs(E[mfe]) * 2 / t_e.size, color="#2E75B6")
ejes[1].axvspan(58, 62, color="#C55A11", alpha=0.25, label="Banda eliminada")
ejes[1].set(title="Espectro: pico de 60 Hz", xlabel="f (Hz)", ylabel="Amplitud"); ejes[1].legend()
ejes[2].plot(t_e[v], ecg[v], color="#2F8F6B", lw=3, alpha=0.5, label="ECG limpio")
ejes[2].plot(t_e[v], ecg_filtrado[v], color="#1F3864", lw=0.9, label="ECG filtrado")
ejes[2].set(title="Resultado del filtrado", xlabel="t (s)"); ejes[2].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(CARPETA / "fig6_aplicacion_ecg.png", dpi=150)
plt.close(fig)

print("\nGráficas guardadas en la carpeta 'figuras/'.")
