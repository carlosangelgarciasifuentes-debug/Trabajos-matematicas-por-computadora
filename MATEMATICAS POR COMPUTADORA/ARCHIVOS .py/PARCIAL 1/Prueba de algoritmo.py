import math
import matplotlib.pyplot as plt
import numpy as np

def calcular_area_del_circulo(radio, unidades):
    """Calcula el área de un círculo dado su radio y las unidades."""
    area = math.pi * radio ** 2
    return f"El área del círculo con radio {radio} {unidades} es: {area:.2f} {unidades}²"

def pedir_datos():
    radio = float(input("Ingrese el radio del círculo: "))
    unidades = input("Ingrese las unidades (ejemplo: cm, m): ")
    resultado = calcular_area_del_circulo(radio, unidades)
    print(resultado)

if __name__ == "__main__": 
    pedir_datos()
