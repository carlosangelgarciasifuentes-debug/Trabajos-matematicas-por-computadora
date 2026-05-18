
"""
Exercise 2.51 

Let f(x) be a continuous function on the interval  [a,b] where f(a)⋅f(b)<0. Clearly give
all of the mathematical details for how the Bisection Method approximates the root of the 
functionf(x) in the interval [a,b].
"""

def metodo_de_la_biseccion(f, a, b, tol=1e-6, max_iter=100): #Agregan las variables

    #Según el algoritmo debe de decinor si hay un cambio de signo en el intervalo

    if f(a)*f(b) >= 0:
        raise ValueError("No hay cambio de signo en el intervalo.")

    # Inicializamos la variable c para almacenar la aproximación de la raíz

    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or abs(b - a) < tol:
            return c
        
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c

    #Se vuelve a generar la iteracion

    return c
