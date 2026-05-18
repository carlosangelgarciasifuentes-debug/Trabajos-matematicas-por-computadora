#PROBLEMAS DEL 5, 6, 9 Y 10

import math
import numpy as np
import matplotlib.pyplot as plt

#Aplique el meotdo de iteración de punto fijo para determinar una solución con una exactitud de 10^-2 para x^4-3x^2-3=0 en
#[0,2]. Utilice P0=1

def punto_fijo_1(p0, tol, max_iter):
    g = lambda x: math.sqrt(3 + 3/(x**2))
    
    print(f"{'Iteración':<12} | {'p_n':>12}")
    print("-"*27)
    
    for i in range(1, max_iter + 1):
        if p0 == 0:
            print("Error: p0 no puede ser 0")
            return None, 0
        
        p = g(p0)
        print(f"{i:<12} | {p:>12.6f}")
        
        if abs(p - p0) < tol:
            print(f"\nRaíz aproximada: {p:.6f}")
            print(f"Iteraciones: {i}")
            return p, i
        
        p0 = p
    
    return p, max_iter

punto_fijo_1(1, 1e-2, 100)

#=========================================================================================================================

#Aplique un metodo de iteración de punto fijo para determinar una solución exacta dentro de 10^-2 para x^3-x-1=0 en 
#[1,2]. Utilice P0=1

def punto_fijo_2(p0, tol, max_iter):
    g = lambda x: (x + 1)**(1/3)
    
    print(f"{'Iteración':<12} | {'p_n':>12}")
    print("-"*27)
    
    for i in range(1, max_iter + 1):
        p = g(p0)
        print(f"{i:<12} | {p:>12.6f}")
        
        if abs(p - p0) < tol:
            print(f"\nRaíz aproximada: {p:.6f}")
            print(f"Iteraciones: {i}")
            return p, i
        
        p0 = p
    
    return p, max_iter

raiz, iteraciones = punto_fijo_2(1, 1e-2, 100)

#=========================================================================================================================

#Aplique un metodo de iteración de punto fijo para obtener una aproximación a raíz de 3 con una exactitud de 10^-4.
#Compare su resultado con el numero de iteraciones que requiere la respuesta obtenida en el ejericio 10 de la seccion 2.1

def raiz_3(p0, tol, max_iter):
    g = lambda x: 0.5 * (x + 3/x)
    
    print(f"{'Iteración':<12} | {'p_n':>12}")
    print("-"*27)
    
    for i in range(1, max_iter + 1):
        p = g(p0)
        print(f"{i:<12} | {p:>12.6f}")
        
        if abs(p - p0) < tol:
            print(f"\nRaíz aproximada: {p:.6f}")
            print(f"Iteraciones: {i}")
            return p, i
        
        p0 = p
    
    return p, max_iter

raiz, iteraciones = raiz_3(1, 1e-4, 100)

#=========================================================================================================================

#Use un metodo de iteración de punto fijo para obtener una aproximación a raíz cubica de 25 con una exactitud de 10^-4.
#Compare su resultado con el número de iteraciones que requiere la respuesta obtenida en el ejercicio 11 de la sección 2.1

def raiz_cubica_25(p0, tol, max_iter):
    g = lambda x: (2*x + 25/(x**2)) / 3
    
    print(f"\n{'Iteración':<12} | {'p_n':>12}")
    print("-"*27)
    
    for i in range(1, max_iter + 1):
        p = g(p0)
        print(f"{i:<12} | {p:>12.6f}")
        
        if abs(p - p0) < tol:
            print("\nRaíz aproximada:", f"{p:.6f}")
            print("Iteraciones:", i)
            return p, i
        
        p0 = p

raiz_cubica_25(3, 1e-4, 100)

#=========================================================================================================================