import numpy as np

def ejercicio_2(F, J, x, imax=100, tol=1e-18):
    cumplimiento_del_ejercicio = False
    k = 0
    while (not cumplimiento_del_ejercicio and k < imax):
        delta_de_x = np.linalg.solve(J(x), -F(x))
        x = x + delta_de_x
        k += 1
        error = np.linalg.norm(F(x))

        print(f"{k:<3} {str(x):<30} {error:<12.4f}")

        cumplimiento_del_ejercicio = np.linalg.norm(F(x)) <= tol
        k += 1
    if k < imax:
        return x
    else:
        raise ValueError('LA FUNCIÓN NO ES POSIBLE QUE CONVERGA EN ESTOS RANGOS DE VALORES')

def f(x):
    return np.array([ 
        x[0]**2 + x[0]*x[1] - 10,
        x[1] + 3*x[0]*(x[1]**2) - 57
    ])

def j(x):
    return np.array([
        [2*x[0] + x[1],  x[0]          ],
        [3*(x[1]**2),    1 + 6*x[0]*x[1]]
    ])

def ejercicio_3():

    puntos = [
        (-2.2047, 5.2202),
        ( 4.7297, 1.2313),
        ( 3.5377, 6.5454)
    ]

    A = []
    b = []

    for x, y in puntos:
        A.append([x, y, 1])
        b.append(-(x**2 + y**2))

    A = np.array(A)
    b = np.array(b)

    D, E, F = np.linalg.solve(A, b)

    a = -D / 2 
    b1 = -E / 2 
    r = np.sqrt(a**2 + b1**2 - F)

    print(f"Centro en x  (a) = {a:.4f}")
    print(f"Centro en y  (b) = {b1:.4f}")
    print(f"Radio        (r) = {r:.4f}")

def main():
    print("EJERCICIO 2 \n")
    x0 = np.array([4.0, 5.0])
    raiz = ejercicio_2(f, j, x0)
    print(f"Raíz encontrada: {raiz}")
    print(f"Verificación f(raíz) = {f(raiz)}")

    print("EJERCICIO 3 \n")
    ejercicio_3()

if __name__ == "__main__":
    main()
