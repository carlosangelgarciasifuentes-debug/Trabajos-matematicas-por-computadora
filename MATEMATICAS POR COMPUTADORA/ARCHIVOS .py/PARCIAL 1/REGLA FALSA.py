import math
import numpy as np
import matplotlib.pyplot as plt


def reglafalsa(f, x0, x1, tol=1e-8):
    if f(x0) * f(x1) > 0:
        raise ValueError("La función no cambia de signo en este intervalo")

    print("{:^5s} {:^10s} {:^10s} {:^10s} {:^10s}".format(
        "i", "x0", "x1", "x", "f(x)"
    ))

    i = 0
    while True:
        x = (x0*f(x1) - x1*f(x0)) / (f(x1) - f(x0))

        print("{:^5d} {:10.5f} {:10.5f} {:10.5f} {:10.5f}".format(
            i, x0, x1, x, f(x)
        ))

        if abs(f(x)) < tol:
            break

        if f(x0) * f(x) < 0:
            x1 = x
        else:
            x0 = x

        i += 1

    return x



def f(x):
    return -0.5*x**2+2.5*x+4.5

def main():
    x0 = 6
    x1 = 7
    tol = 1e-6

    raiz = reglafalsa(f, x0, x1, tol)
    print("\nRaíz aproximada:", raiz)

    x=np.linspace(x0,x1,200)
    y=f(x)

    fig=plt.figure()
    plt.plot(x,y)
    plt.scatter(raiz, f(raiz))
    plt.text(raiz, f(raiz), ' Raiz ' + str(raiz), color = 'red')
    plt.grid()

    plt.show()

if __name__ == "__main__":


    main()
