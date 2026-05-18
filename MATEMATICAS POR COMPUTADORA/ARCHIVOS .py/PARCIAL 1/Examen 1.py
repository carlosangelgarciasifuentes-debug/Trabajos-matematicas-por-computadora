#La ecuacion x**3-1.2x**2-8.19x+13.23 tiene una raíz doble cerca de x =2.
#Diseña un codigo que utilice el método de Newthon Raphson para encontrar estas raíces.

import math
import numpy as np 
import matplotlib.pyplot as np

#==========Función del codigo==========

def f(x):
    return x**3 - 1.2*x**2 - 8.19*x + 13.23

def df(x):
    return 3*x**2 - 2.4*x - 8.19

#50 iteraciones como valor
#tolerancia de 1e-4 como valor

def funcion_del_ejercicio():
    x0 = 2.0
    tol = 1e-4
    max_iter = 50

    for i in range(max_iter):
        fx = f(x0)
        dfx = df(x0)

        if abs(dfx) < 1e-12:
            print("Derivada cercana a cero, no se puede continuar.")
            return None

        x1 = x0 - fx / dfx
        tabla_iteraciones(i+1, x1, f(x1))

        if abs(f(x1)) < tol:
            return x1

        x0 = x1

    print("No se encontró una raíz dentro de la tolerancia después de 50 iteraciones.")
    return None

#Imprimir en columnas i --> itercion, x_i --> valor actual de x. f(x_i) --> evaluación del valor actual de x.

def tabla_iteraciones(i, x_i, fx_i):
    print(f"{i:>5} | {x_i:>10.6f} | {fx_i:>12.6e}")

# Sacar la raíz al final del ejercicio y mostrarla al usuario.
print("________________________________________")
print("Iteración |     x_i     |    f(x_i)   ")
print("________________________________________")
raiz = funcion_del_ejercicio()
if raiz is not None:
    print(f"\nRaíz aproximada: {raiz:.6f}")
