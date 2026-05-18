#Representar A como array en una variable, donde se encuentra su inversa multiplicandolo por el vector de respuestas que es B
#Siendo estos la respuesta del vector de incognitas donde será:
#X = [x1,x2,x3,x4,x5]
#Siendo que X= A^-1 * B
import numpy as np
from fractions import Fraction

A = np.array([[5, -1, 4, 7, 8], [-3, 0, 2, -6, 1],[9, -2, -4, 3, 5],[7, 6, -8, -7, 0],[2, 0, 1, 4, 9]],
    dtype=float)

# VECTOR B
B = np.array([20, 13, 2, 7, 11], dtype=float)

# Calcular X = A⁻¹ × B
A_inversa = np.linalg.inv(A)
X = np.dot(A_inversa, B)

# Convertir a fracciones
X_fracciones = [Fraction(x).limit_denominator(10000) for x in X]

# Mostrar resultados
print("╔" + "═"*60 + "╗")
print("║" + " X = A⁻¹ × B ".center(60) + "║")
print("╚" + "═"*60 + "╝\n")

print("VECTOR X:")
for i, x in enumerate(X_fracciones):
    print(f"x{i+1} = {x}")
print(f"\nX = {X_fracciones}")

# Verificar: A × X = B
print("\n" + "═"*60)
print("VERIFICACIÓN: A × X = B\n")
AX = np.dot(A, X)
for i in range(len(B)):
    print(f"Ecuación {i+1}: {AX[i]:.6f} = {B[i]} ✓")