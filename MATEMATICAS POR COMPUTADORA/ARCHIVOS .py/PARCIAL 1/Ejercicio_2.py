import numpy as np 
import matplotlib.pyplot as plt
import math
import cmath

def newtonRaphson(f, df, x, i, i_max, tol=1e-8):
    k = 0
    cumple = False

print("{:^10s}{:^10s}{:^10s}".format( "x", "f(x)", "df(x)"))
while (not cumple and k < i_max):
    if df(x) != 0:
        x = x - f(x)/df(x)
    else:
        x=x + tol
    print("{:10.5fs}{:10.5f}{:10.5f}".format( x, f(x), df(x)))
    if abs(f(x)) < tol:
        cumple = abs(f(x)) <= tol
    k += 1

if k < i_max:
    return x
else:
    raise ValueError("La función no converge")

def f(x):
    R=10
    return math.pi * x ** 2 * (3*R-x)/3-800 

def df(x):
    return math.pi*x**2

def main():
    #valores iniciales
    x0=10
    #Llamada al algoritmo
    raiz = newtonRaphson(f,df,x0)
    print("f({:e}) = {:e}".format(raiz,f(raiz)))

main()



