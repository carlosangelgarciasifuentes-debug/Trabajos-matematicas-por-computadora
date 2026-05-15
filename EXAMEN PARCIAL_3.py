import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    i1, i3 = y
    # Coeficientes exactos según la imagen del ejercicio
    di1dt = -20*i1 + 10*i3 + 100  # <--- Aquí estaba el error (era -20, no -10)
    di3dt =  10*i1 - 20*i3
    return np.array([di1dt, di3dt])

def rk4(f, y0, t0, tf, h):
    t_values = np.linspace(t0, tf, int((tf - t0) / h) + 1)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        k1 = f(t,       y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h,   y + h   * k3)
        y_values[i] = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    return t_values, y_values

# Condiciones iniciales i1(0)=0, i3(0)=0
y0 = [0.0, 0.0]
t0, tf, h = 0, 5, 0.01

t_values, y_values = rk4(f, y0, t0, tf, h)

# Tabla para hoja de máquina
print(f"{'t':>6} {'i1(t)':>12} {'i3(t)':>12}")
print("-" * 32)
for i, t in enumerate(t_values):
    if any(abs(t - v) < 1e-9 for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        print(f"{t:>6.2f} {y_values[i,0]:>12.6f} {y_values[i,1]:>12.6f}")

# Gráfica
i_R = y_values[:, 0] - y_values[:, 1]

plt.figure(figsize=(10, 5))
plt.plot(t_values, y_values[:, 0], label='$i_1(t)$ (Corriente en $L_1$)', color='blue')
plt.plot(t_values, y_values[:, 1], label='$i_3(t)$ (Corriente en $L_2$)', color='red')
plt.plot(t_values, i_R,            label='$i_R(t)$ (Corriente en $R_1, i_2$)', 
         color='green', linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('Corriente (A)')
plt.title('Detalle de la Fase Transitoria (0 s a 5 s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()