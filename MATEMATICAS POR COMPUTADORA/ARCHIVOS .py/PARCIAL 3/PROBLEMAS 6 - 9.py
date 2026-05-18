import numpy as np
import matplotlib.pyplot as plt

def derivative_formulas():
    # --- Datos Problema 6 (h = 0.01) ---
    # Usando Forward Difference O(h^2) porque 2.36 está al inicio
    x6 = np.array([2.36, 2.37, 2.38, 2.39])
    f6 = np.array([0.85866, 0.86289, 0.86710, 0.87129])
    h6 = 0.01

    # f'(x0) = (-3f0 + 4f1 - f2) / (2h)
    df6 = (-3*f6[0] + 4*f6[1] - f6[2]) / (2 * h6)
    
    # f''(x0) = (2f0 - 5f1 + 4f2 - f3) / h^2
    ddf6 = (2*f6[0] - 5*f6[1] + 4*f6[2] - f6[3]) / (h6**2)

    print(f"Problema 6: f'(2.36) ≈ {df6:.4f}, f''(2.36) ≈ {ddf6:.4f}")

    # --- Datos Problema 7 (h = 0.05) ---
    # Central Difference O(h^2) para el punto central x=1.00
    f8 = np.array([0.431711, 0.398519, 0.367879, 0.339596, 0.313486])
    h8 = 0.05
    # f''(x) = (f(x-h) - 2f(x) + f(x+h)) / h^2 -> f8[1], f8[2], f8[3]
    ddf7 = (f8[1] - 2*f8[2] + f8[3]) / (h8**2)
    
    print(f"\nProblema 7: f''(1.00) ≈ {ddf7:.6f}")

    # --- Datos Problema 8 (h = 0.08) ---
    # Central Difference O(h^2) para el punto central x=1.00
    f8 = np.array([0.431711, 0.398519, 0.367879, 0.339596, 0.313486])
    h8 = 0.08
    # f''(x) = (f(x-h) - 2f(x) + f(x+h)) / h^2 -> f8[1], f8[2], f8[3]
    ddf8 = (f8[1] - 2*f8[2] + f8[3]) / (h8**2)
    
    print(f"\nProblema 8: f''(1.00) ≈ {ddf8:.6f}")

    # --- Datos Problema 9 (h = 0.1) ---
    # Five-Point Central Difference para f'(0.2)
    f9 = np.array([0.000000, 0.078348, 0.138910, 0.192916, 0.244981])
    h9 = 0.1
    # Formula: (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    df9 = (-f9[4] + 8*f9[3] - 8*f9[1] + f9[0]) / (12 * h9)

    print(f"\nProblema 9: f'(0.2) ≈ {df9:.6f}")

if __name__ == "__main__":
    derivative_formulas()