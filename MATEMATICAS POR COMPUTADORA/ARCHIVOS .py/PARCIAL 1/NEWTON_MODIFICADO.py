#MODIFICACIÓN DEL MÉTODO DE NEWTON PARA RESOLVER FUNCIONES 

import argparse
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

 #PERMITIR QUE EL USUARIO INGRESE LA FUNCIÓN, EL VALOR INICIAL, LA TOLERANCIA Y EL NÚMERO MÁXIMO DE ITERACIONES.

def ModificatenewtonRaphson(f, df, d2f, x, imax=100,tol=1e-8):
    cumple=False
    print('{:^10s}{:^10s}{:^10s}'.format('x','f(x)','df(x)','d2f(x)'))

    k=0
    while (not cumple) and (k<imax):
        x = x - (f(x)*df(x))/(df(x)**2 - f(x)*d2f(x))
        print('{:^10.5f}{:^10.5f}{:^10.5f}{:^10.5f}'.format(x,f(x),df(x),d2f(x)))
        cumple=abs(f(x))<tol
        k+=1

    if k<imax:
        return x
    else:
        raise ValueError("LA FUNCION NO CONVERGE")


# funcion a evaluar
def f(x):
    return (x-2)*(x-2)*(x-4)

# primera derivada
def df(x):
    return (x-4)*(2*x-4) + (x-2)**2

# segunda derivada
def d2f(x):
    return 2*(3*x-8)

def main():
    #valores iniciales
    x0 = 1
    #Llamada del algoritmo
    raiz = ModificatenewtonRaphson(f, df, d2f, x0)
    print('f({:e})={:e}'.format(raiz,f(raiz)))

    x=np.linspace(0,5,200)
    y=f(x)

    fig=plt.figure()
    plt.plot(x,y)
    plt.title('$f(x)=(x-2)^2(x-4)$')
    plt.scatter(raiz, f(raiz))
    plt.text(raiz, f(raiz), ' Raiz ' + str(raiz), color = 'red')
    plt.grid()

    plt.show()
    fig.savefig("newton_modificado.pdf", bbox_inches='tight')

if __name__ == "__main__": main()