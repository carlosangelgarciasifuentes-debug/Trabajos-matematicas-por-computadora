import numpy as np
import matplotlib.pyplot as plt

# 1. Definición de la función externa f(t, x)
def f(t, x):
    return t + x

# --- INICIO DEL ALGORITMO DE EULER ---

# 1. Representar variables
n = 100  
a = 1.0
b = 2.0
x_0 = -4.0  # Guardamos el valor inicial
h = (b - a) / n

# Listas para almacenar resultados para graficar
t_euler = [a]
x_euler = [x_0]

# 2. Inicialización de variables de iteración
t = a
x = x_0

print(f"{'k':>5} | {'t':>10} | {'x':>10}")
print("-" * 30)
print(f"{0:>5} | {t:>10.4f} | {x:>10.4f}")

# Para k = 1 hasta n
for k in range(1, n + 1):
    # Algoritmo de Euler
    x = x + h * f(t, x)
    t = t + h
    
    # Guardar valores
    t_euler.append(t)
    x_euler.append(x)
    
    # Imprimir cada 10 pasos para no saturar la consola
    if k % 10 == 0:
        print(f"{k:>5} | {t:>10.4f} | {x:>10.4f}")

# --- GRAFICACIÓN ---

# La solución exacta para dx/dt - x = t con x(1) = -4 es:
# x(t) = -t - 1 - 2 * exp(t - 1)
def exact_solution(t):
    return -t - 1 - 2 * np.exp(t - 1)

t_values = np.linspace(a, b, 200)
x_exact = exact_solution(t_values)

plt.figure(figsize=(10, 6))
plt.plot(t_values, x_exact, label='Solución Exacta (Analítica)', color='blue', lw=2)
plt.plot(t_euler, x_euler, 'r--', label=f'Aproximación Euler (n={n})', alpha=0.8)

plt.title('Método de Euler vs Solución Exacta')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- TERMINAR PROBLEMA DE EULER ---

"""
def euler(n=100, a=0, b=0, xprima):
    h=(b-a)/n
    t=a
    for k in range (1,n+1)
    printf (ejemplo)
"""