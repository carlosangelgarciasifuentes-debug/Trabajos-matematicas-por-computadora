"""
=============================================================
  MATEMÁTICAS POR COMPUTADORA — UASLP FEPZM
  Ejercicios 67 al 80: Mínimos Cuadrados e Interpolación
  Alumno : Carlos Ángel García Sifuentes
  Docente: Prof. Ing. Jesús Padrón
=============================================================
"""

import numpy as np

SEP = "=" * 60

# ─────────────────────────────────────────────────────────────
# EJERCICIO 67
# Demostración: (X̄, Ȳ) pertenece a la recta de mínimos cuadrados
# ─────────────────────────────────────────────────────────────
print(SEP)
print("EJERCICIO 67")
print("Demostración: (X̄, Ȳ) está en la recta de mínimos cuadrados")
print(SEP)

x = np.array([1.2, 3.1, 5.6, 6.2, 8.8, 9.1])
y = np.array([4.957, 12.909, 23.404, 25.981, 36.907, 38.212])
n = len(x)

Xbar, Ybar = np.mean(x), np.mean(y)
b = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / \
    (n*np.sum(x**2) - np.sum(x)**2)
a = (np.sum(y) - b*np.sum(x)) / n

y_en_Xbar = a + b * Xbar
print(f"  Recta ajustada : y = {a:.4f} + {b:.4f}·x")
print(f"  y(X̄) = {y_en_Xbar:.6f}")
print(f"  Ȳ    = {Ybar:.6f}")
print(f"  ¿Son iguales? {np.isclose(y_en_Xbar, Ybar)}")  # True

# ─────────────────────────────────────────────────────────────
# EJERCICIO 68
# Ajuste lineal bidireccional
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 68")
print("Ajuste lineal bidireccional")
print(SEP)

x = np.array([1.2, 3.1, 5.6, 6.2, 8.8, 9.1])
y = np.array([4.957, 12.909, 23.404, 25.981, 36.907, 38.212])
n = len(x)

# Parte a) x libre de errores
b_a = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / \
      (n*np.sum(x**2) - np.sum(x)**2)
a_a = (np.sum(y) - b_a*np.sum(x)) / n
SS_a = np.sum((y - (a_a + b_a*x))**2)
print(f"  a) y = {a_a:.4f} + {b_a:.4f}·x   SS = {SS_a:.6f}")

# Parte b) y libre de errores
bb = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / \
     (n*np.sum(y**2) - np.sum(y)**2)
a_b = (np.sum(x) - bb*np.sum(y)) / n
slope     = 1.0 / bb
intercept = -a_b / bb
SS_b = np.sum((x - (a_b + bb*y))**2)
print(f"  b) y = {intercept:.4f} + {slope:.4f}·x   SS = {SS_b:.6f}")
print(f"  c) Pendientes iguales: {np.isclose(b_a, slope, atol=1e-2)}")
print(f"  d) Menor SS vertical: {SS_a:.6f}  |  Menor SS horizontal: {SS_b:.6f}")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 69
# Ajuste de plano z = ax + by + c
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 69")
print("Ajuste de plano z = ax + by + c")
print(SEP)

x = np.array([0.40, 1.2, 3.4, 4.1, 5.7, 7.2, 9.3])
y = np.array([0.70, 2.1, 4.0, 6.3, 6.3, 8.1, 8.9])
z = np.array([0.031, 0.933, 3.058, 3.349, 4.870, 5.757, 8.921])

A   = np.column_stack([x, y, np.ones(len(x))])
AtA = A.T @ A
Atz = A.T @ z

print("  a) Ecuaciones normales — Matriz AtA:")
print(np.round(AtA, 4))

coef = np.linalg.solve(AtA, Atz)
a_c, b_c, c_c = coef
print(f"\n  b) z = {a_c:.4f}·x + {b_c:.4f}·y + {c_c:.4f}")

z_hat = A @ coef
SS = np.sum((z - z_hat)**2)
print(f"  c) Suma de cuadrados SS = {SS:.6f}")

print("\n  Valores ajustados vs reales:")
print(f"  {'z_real':>8} {'z_ajust':>9} {'error':>8}")
for i in range(len(z)):
    print(f"  {z[i]:>8.3f} {z_hat[i]:>9.4f} {z[i]-z_hat[i]:>8.5f}")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 70
# Mínimos cuadrados con tercer punto en x = 4
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 70")
print("Mínimos cuadrados con tercer punto en x = 4")
print(SEP)

def ls_line(xs, ys):
    n = len(xs)
    b = (n*np.sum(xs*ys) - np.sum(xs)*np.sum(ys)) / \
        (n*np.sum(xs**2) - np.sum(xs)**2)
    a = (np.sum(ys) - b*np.sum(xs)) / n
    return a, b

casos = [("a) y3=5",  5.0),
         ("b) y3=0",  0.0),
         ("c) y3=4",  4.0),
         ("d) y3=-1", -1.0)]

print(f"  {'Caso':<12} {'a':>8} {'b':>8} {'SS':>10}")
for label, y3 in casos:
    xs = np.array([2., 4., 6.])
    ys = np.array([5., y3, -1.])
    a, b = ls_line(xs, ys)
    SS = np.sum((ys - (a + b*xs))**2)
    print(f"  {label:<12} {a:>8.4f} {b:>8.4f} {SS:>10.4f}")

print("  → La pendiente siempre es -1.5; solo cambia el intercepto.")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 71
# Ajuste exponencial S = A·e^(BT)
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 71")
print("Ajuste exponencial S = A·exp(BT)")
print(SEP)

T = np.array([77., 100., 185., 239., 285.])
S = np.array([2.4, 3.4, 7.0, 11.1, 19.6])
n = len(T)
Y = np.log(S)

B   = (n*np.sum(T*Y) - np.sum(T)*np.sum(Y)) / \
      (n*np.sum(T**2) - np.sum(T)**2)
lnA = (np.sum(Y) - B*np.sum(T)) / n
A   = np.exp(lnA)
print(f"  S = {A:.4f} * exp({B:.6f} * T)")

print(f"\n  {'T':>6} {'S_real':>8} {'S_ajust':>9} {'error':>8}")
for i in range(n):
    S_hat = A * np.exp(B * T[i])
    print(f"  {T[i]:>6.0f} {S[i]:>8.1f} {S_hat:>9.4f} {S[i]-S_hat:>8.4f}")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 72
# Análisis gráfico: R² lineal vs semi-logarítmico
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 72")
print("Comparación R²: escala lineal vs semi-logarítmica")
print(SEP)

T   = np.array([77., 100., 185., 239., 285.])
S   = np.array([2.4, 3.4, 7.0, 11.1, 19.6])
lnS = np.log(S)

r2_lin = np.corrcoef(T, S)[0, 1]**2
r2_log = np.corrcoef(T, lnS)[0, 1]**2
print(f"  R² escala lineal  : {r2_lin:.4f}")
print(f"  R² escala semi-log: {r2_log:.4f}  ← mejor ajuste")
print("  → El modelo exponencial S = A·exp(BT) es el apropiado.")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 73
# Comparación: ajuste lineal, cuadrático y cúbico
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 73")
print("Comparación de ajustes: lineal, cuadrático y cúbico")
print(SEP)

np.random.seed(42)
x = np.sort(np.random.uniform(2, 7, 6))
z = np.random.uniform(0, 0.2, 6)
y = x**2 + z

print(f"  x (aleatorios en [2,7]): {np.round(x, 4)}")
print(f"  z (ruido en [0,0.2])   : {np.round(z, 4)}")
print(f"  y = x² + z             : {np.round(y, 4)}")
print()

prev_ss = None
for deg in [1, 2, 3]:
    A    = np.column_stack([x**i for i in range(deg + 1)])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    SS   = np.sum((y - A @ coef)**2)
    delta = f"  Δ = {prev_ss - SS:.8f}" if prev_ss else ""
    print(f"  Grado {deg}: SS = {SS:.8f}{delta}")
    prev_ss = SS

print("  → Grado 2 es óptimo (datos son exactamente cuadráticos + ruido).")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 74
# Matriz de diseño A y ecuaciones normales
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 74")
print("Matriz de diseño A y ecuaciones normales")
print(SEP)

x = np.array([1.2, 3.1, 5.6, 6.2, 8.8, 9.1])
y = np.array([4.957, 12.909, 23.404, 25.981, 36.907, 38.212])
n = len(x)

A   = np.column_stack([np.ones(n), x])
AtA = A.T @ A
Aty = A.T @ y

manual_AtA = np.array([[n,         np.sum(x)],
                        [np.sum(x), np.sum(x**2)]])
manual_Aty = np.array([np.sum(y), np.sum(x*y)])

print("  a) A^T·A =")
print(f"     {np.round(AtA, 4)}")
print(f"     Coincide con sumas manuales: {np.allclose(AtA, manual_AtA)}")

print("\n  b) A^T·y =", np.round(Aty, 4))
print(f"     Coincide con sumas manuales: {np.allclose(Aty, manual_Aty)}")

coef = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"\n  c) Solución con lstsq (QR/SVD, más estable):")
print(f"     a = {coef[0]:.4f},  b = {coef[1]:.4f}")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 75
# Ajuste potencial F = k·P^n
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 75")
print("Ajuste potencial F = k·P^n")
print(SEP)

P = np.array([10., 16., 25., 40., 60.])
F = np.array([94., 118., 147., 180., 230.])
N = len(P)
lnP, lnF = np.log(P), np.log(F)

n_exp = (N*np.sum(lnP*lnF) - np.sum(lnP)*np.sum(lnF)) / \
        (N*np.sum(lnP**2) - np.sum(lnP)**2)
lnk = (np.sum(lnF) - n_exp*np.sum(lnP)) / N
k   = np.exp(lnk)
r2  = np.corrcoef(lnP, lnF)[0, 1]**2

print(f"  F = {k:.4f} * P^{n_exp:.4f}")
print(f"  R² en log-log: {r2:.6f}")
print(f"  → El exponente ≈ 0.5 confirma F ∝ √P")

print(f"\n  {'P':>6} {'F_real':>8} {'F_ajust':>9} {'error%':>8}")
for i in range(N):
    F_hat = k * P[i]**n_exp
    print(f"  {P[i]:>6.0f} {F[i]:>8.1f} {F_hat:>9.3f} {100*(F[i]-F_hat)/F[i]:>7.2f}%")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 76
# Ajuste cuadrático F = aP² + bP + c
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 76")
print("Ajuste cuadrático F = aP² + bP + c")
print(SEP)

P = np.array([10., 16., 25., 40., 60.])
F = np.array([94., 118., 147., 180., 230.])

A    = np.column_stack([P**2, P, np.ones(len(P))])
coef = np.linalg.lstsq(A, F, rcond=None)[0]
a_q, b_q, c_q = coef
F_hat   = A @ coef
SS_quad = np.sum((F - F_hat)**2)

print(f"  a) F = {a_q:.6f}·P² + {b_q:.4f}·P + {c_q:.4f}")
print(f"     SS cuadrático = {SS_quad:.4f}")

k_75, n_75 = 38.66, 0.4764
SS_pot = np.sum((F - k_75 * P**n_75)**2)
print(f"\n  b) SS potencial (ej. 75) = {SS_pot:.4f}")
winner = "cuadrático" if SS_quad < SS_pot else "potencial"
print(f"     → Menor SS: modelo {winner}")
print("     → Pero el modelo potencial es más apropiado físicamente.")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 77
# Polinomio de grado óptimo
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 77")
print("Polinomio de grado óptimo")
print(SEP)

x  = np.array([1.1, 1.6, 11.4, 4.1, 5.3, 17.5, 9.4, 11.5, 12.1])
fx = np.array([7.9, 24.8, -28.8, 42.6, 29.6, -34.6, -3.1, -28.7, -39.6])

print(f"  {'Grado':>6}  {'SS':>14}  {'Mejora':>14}")
prev_ss = None
for deg in range(1, 8):
    A    = np.column_stack([x**i for i in range(deg + 1)])
    coef = np.linalg.lstsq(A, fx, rcond=None)[0]
    SS   = np.sum((fx - A @ coef)**2)
    delta = f"{prev_ss - SS:.4f}" if prev_ss else "---"
    print(f"  {deg:>6}  {SS:>14.4f}  {delta:>14}")
    prev_ss = SS

print("  → Grado 4 o 5 es el óptimo (mayor mejora antes del sobreajuste).")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 78
# Repetición con subconjuntos de puntos
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 78")
print("Repetición del Ej. 77 con subconjuntos de puntos")
print(SEP)

x  = np.array([1.1, 1.6, 11.4, 4.1, 5.3, 17.5, 9.4, 11.5, 12.1])
fx = np.array([7.9, 24.8, -28.8, 42.6, 29.6, -34.6, -3.1, -28.7, -39.6])

def poly_ss(xs, ys, label, max_deg=5):
    print(f"\n  {label}  x = {np.round(xs, 1)}")
    for deg in range(1, min(len(xs), max_deg + 1)):
        A    = np.column_stack([xs**i for i in range(deg + 1)])
        coef = np.linalg.lstsq(A, ys, rcond=None)[0]
        SS   = np.sum((ys - A @ coef)**2)
        print(f"    Grado {deg}: SS = {SS:.4f}")

poly_ss(x[::2],  fx[::2],  "a) Índices pares  (0,2,4,6,8):")
poly_ss(x[1::2], fx[1::2], "b) Índices impares (1,3,5,7):")
print("\n  c) Los subconjuntos dan coeficientes distintos al ajuste completo.")
print("     Usar todos los datos siempre es más confiable.")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 79
# Ajuste f(x) = a + b·sin(c·x)
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 79")
print("Ajuste f(x) = a + b·sin(c·x)")
print(SEP)

x  = np.array([1.1, 1.6, 11.4, 4.1, 5.3, 17.5, 9.4, 11.5, 12.1])
fx = np.array([7.9, 24.8, -28.8, 42.6, 29.6, -34.6, -3.1, -28.7, -39.6])

print("  a) Con c desconocido el modelo es no lineal → se necesita")
print("     optimización iterativa (ej. Levenberg-Marquardt).")

print("\n  b-c) Con c = π/10 conocido: u_i = sin(π/10·x_i)")
c = np.pi / 10
u = np.sin(c * x)
n = len(x)

b_79 = (n*np.sum(u*fx) - np.sum(u)*np.sum(fx)) / \
       (n*np.sum(u**2) - np.sum(u)**2)
a_79 = (np.sum(fx) - b_79*np.sum(u)) / n
SS   = np.sum((fx - (a_79 + b_79*u))**2)

print(f"     a = {a_79:.4f},  b = {b_79:.4f}")
print(f"     f(x) = {a_79:.4f} + {b_79:.4f}·sin(π/10·x)")
print(f"     SS   = {SS:.4f}")

print(f"\n  {'x':>6} {'f_real':>8} {'f_ajust':>9} {'error':>8}")
for i in range(n):
    f_hat = a_79 + b_79*u[i]
    print(f"  {x[i]:>6.1f} {fx[i]:>8.1f} {f_hat:>9.4f} {fx[i]-f_hat:>8.4f}")

# ─────────────────────────────────────────────────────────────
# EJERCICIO 80
# Número de condición de la matriz para P_n(x)
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EJERCICIO 80")
print("Condicionamiento de la matriz para P_n(x) en [3, 7]")
print(SEP)

x = np.linspace(3, 7, 10)
print(f"  {'Grado':>7}  {'Cond(AtA)':>16}  {'log10(cond)':>12}")
for deg in [4, 6, 8]:
    A    = np.column_stack([x**i for i in range(deg + 1)])
    AtA  = A.T @ A
    cond = np.linalg.cond(AtA)
    print(f"  P_{deg}(x):  {cond:>16.2e}  {np.log10(cond):>12.2f}")

print("\n  Comparación de estabilidad numérica (grado 8):")
deg    = 8
A      = np.column_stack([x**i for i in range(deg + 1)])
y_test = x**3 - 2*x + 1

try:
    c_s = np.linalg.solve(A.T @ A, A.T @ y_test)
    print(f"  SS con solve (AtA): {np.sum((y_test - A@c_s)**2):.4e}")
except np.linalg.LinAlgError:
    print("  solve: matriz singular (demasiado mal condicionada)")

c_l = np.linalg.lstsq(A, y_test, rcond=None)[0]
print(f"  SS con lstsq:       {np.sum((y_test - A@c_l)**2):.4e}  ← más estable")
print("  → Para grados > 3-4 siempre usar lstsq en lugar de solve.")

print(f"\n{SEP}")
print("FIN — Ejercicios 67 al 80 completados.")
print(SEP)