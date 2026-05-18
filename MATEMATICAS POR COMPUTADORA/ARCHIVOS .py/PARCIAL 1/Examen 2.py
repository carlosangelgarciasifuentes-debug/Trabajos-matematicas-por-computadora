#Diseña un código con el metodo de la secante ára determinar las dos raíces de:
#sinx - 3*cosx - 2 = 0 que se encuentra en el intervao de (-2,2)

#Xi1=Xi-((Xi-Xi0)/(f(Xi)-f(X0)))*f(Xi)

import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return math.sin(x) - 3*math.cos(x) - 2

#Usar metodo de la secante

def metodo_de_secante(x0, x1, tolerance=1e-4, max_iter=100):
    for _ in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        if abs(fx1 - fx0) < 1e-10:

            break

        x_new = x1 - (fx1 * (x1 - x0)) / (fx1 - fx0)
        if abs(x_new - x1) < tolerance:
            return x_new
        x0 = x1

        x1 = x_new

    return x1



#Determinar las dos raíces de la ecuacion en el intervalo (-2, 2)

raíz1 = metodo_de_secante(-1.5, -1.0)
raíz2 = metodo_de_secante(0.5, 1.5)

print(f"Raíz 1: {raíz1}, f(raíz1) = {f(raíz1)}")
print(f"Raíz 2: {raíz2}, f(raíz2) = {f(raíz2)}")
