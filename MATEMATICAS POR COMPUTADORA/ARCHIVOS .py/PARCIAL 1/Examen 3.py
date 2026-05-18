#Un servomotor controla la posición angular de un brazo mecanico unido a un resorte torsional no lineal
#El equilibrio del sistema se alcanza cuando el par del motor es igual al par del resorte
#El modelo del sistema está dado por la ecuacion
# teta = M/(K+sin(theta)) donde teta es el angulo del equilibrio, M= 3N*m, k = 1.5 N*m 

import math
import numpy as np
import matplotlib.pyplot as plt

def angulo_de_equilibrio():
    M = 3.0
    K = 1.5

    def f(theta):
        return theta - M / (K + math.sin(theta))

    # Usar el método del PUNTO FIJO para encontrar el ángulo de equilibrio

    theta = 0 
    tolerance = 1e-4
    max_iterations = 50
    
    for i in range(max_iterations):

        theta_new = M / (K + math.sin(theta))
        fx_i = theta_new - M / (K + math.sin(theta_new))

        print(f"{i+1:>5} | {theta_new:>10.6f} | {fx_i:>12.6e}")

        if abs(theta_new - theta) < tolerance:

            print(f"Solución encontrada en iteración {i+1}: θ = {theta_new:.6f} radianes")
            return theta_new
        
        theta = theta_new 

# Sacar la raíz al final del ejercicio y mostrarla al usuario, agregar una tabla donde aparezca: 
#iteración, valor de theta, evaluación del valor de theta

print("________________________________________")
print("Iteración |     theta_i     |    f(theta_i)   ")
print("________________________________________")
raiz = angulo_de_equilibrio()
if raiz is not None:
    print(f"\nRaíz aproximada: {raiz:.6f}")