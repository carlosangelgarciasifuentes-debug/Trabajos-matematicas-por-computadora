import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# MÉTODO GENERAL RUNGE-KUTTA DE ORDEN 4
# =========================================================

def rk4(f, t, y, h):
    k1 = np.array(f(t, y))
    k2 = np.array(f(t + h/2, y + h*k1/2))
    k3 = np.array(f(t + h/2, y + h*k2/2))
    k4 = np.array(f(t + h, y + h*k3))

    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

#==========================================================================
#TODOS LOS EJERCICIOS DE LA PAGINA 266 TODA LA SECCIÓN EN NÚMEROS IMPARES
#==========================================================================

print("\n========== EJERCICIO 1 ==========")

def ejercicio_1():

    f = lambda x, y: x**2 - 4*y

    # y''
    df = lambda x, y: 2*x - 4*(x**2 - 4*y)

    x = 0
    y = 1

    h = 0.05

    print(f"{'Paso':<10}{'x':<10}{'y':<15}")

    for i in range(2):

        yp = f(x, y)
        ypp = df(x, y)

        y = y + h*yp + (h**2/2)*ypp
        x += h

        print(f"{i+1:<10}{x:<10.2f}{y:<15.8f}")

    print("\ny(0.1) =", y)

ejercicio_1()

print("\n========== EJERCICIO 3 ==========")

def ejercicio_3():

    f = lambda x, y: np.sin(y)

    # y'' = cos(y)*sin(y)
    df = lambda x, y: np.cos(y)*np.sin(y)

    x = 0
    y = 1

    h = 0.1

    xs = [x]
    ys = [y]

    while x < 0.5:

        yp = f(x, y)
        ypp = df(x, y)

        y = y + h*yp + (h**2/2)*ypp
        x += h

        xs.append(x)
        ys.append(y)

    print(f"Valor final: y({x}) = {y}")

    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.title("Ejercicio 3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

ejercicio_3()

print("\n========== RESULTADOS EJERCICIO 5 ==========")
def ejercicio_5(x, Y):
    y1, y2 = Y
    dy1 = y2
    dy2 = (x * y2 + 2 * y1**2) / y1
    return [dy1, dy2]

print("\n" + "="*50)
print("Sistema de ecuaciones de primer orden para el inciso (b):")
print(f"""
    y1' = y2
    y2' = (x * y2 + 2 * y1^2) / y1
    
    Donde:
    y1 representa a 'y'
    y2 representa a 'y''
""")
print("="*50)

print("\n========== EJERCICIO 7 ==========")

def ejercicio_7():

    g = 9.80665
    L = 1

    def sistema(t, Y):

        theta = Y[0]
        omega = Y[1]

        dtheta = omega
        domega = -(g/L)*np.sin(theta)

        return np.array([dtheta, domega])

    h = 0.01
    t = 0

    Y = np.array([1.0, 0.0])

    ts = []
    angulos = []

    while t <= 10:

        ts.append(t)
        angulos.append(Y[0])

        Y = rk4(sistema, t, Y, h)
        t += h

    # período aproximado teórico
    T = 2*np.pi*np.sqrt(L/g)

    print("Periodo aproximado:", T)

    plt.figure()
    plt.plot(ts, angulos)
    plt.title("Péndulo simple")
    plt.xlabel("t")
    plt.ylabel("θ(t)")
    plt.grid()

ejercicio_7()

print("\n========== EJERCICIO 9 ==========")

def ejercicio_9():

    m = 2.5
    k = 75

    def sistema(t, Y):

        y = Y[0]
        v = Y[1]

        if t < 2:
            P = 10*t
        else:
            P = 20

        dy = v
        dv = (P - k*y)/m

        return np.array([dy, dv])

    h = 0.01
    t = 0

    Y = np.array([0.0, 0.0])

    ts = []
    posiciones = []

    ymax = 0

    while t <= 10:

        ts.append(t)
        posiciones.append(Y[0])

        ymax = max(ymax, abs(Y[0]))

        Y = rk4(sistema, t, Y, h)
        t += h

    print("Desplazamiento máximo:", ymax)

    plt.figure()
    plt.plot(ts, posiciones)
    plt.title("Sistema masa-resorte")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.grid()

ejercicio_9()

print("\n========== EJERCICIO 11 ==========")
 
def ejercicio_11():
    g = 9.80665
    L = 1.0
    Y = 0.25
    omega = 2.5
 
    def sistema(t, estado):
        y1, y2 = estado
        dy1 = y2
        dy2 = -(g / L) * np.sin(y1) + (omega**2 / L) * Y * np.cos(y1) * np.sin(omega * t)
        return [dy1, dy2]
 
    h = 0.01
    t = 0.0
    estado = np.array([0.0, 0.0])  # θ(0) = 0, θ'(0) = 0
 
    ts = []
    thetas = []
 
    while t <= 10.0:
        ts.append(t)
        thetas.append(estado[0])
        estado = rk4(sistema, t, estado, h)
        t += h
 
    theta_max = max(np.abs(thetas))
    print(f"Ángulo máximo θ durante [0, 10] s: {theta_max:.6f} rad  ({np.degrees(theta_max):.4f}°)")
 
    plt.figure()
    plt.plot(ts, np.degrees(thetas), color='steelblue')
    plt.title("Ejercicio 11 — Péndulo con collarín deslizante")
    plt.xlabel("t (s)")
    plt.ylabel("θ (°)")
    plt.grid(True)
    plt.tight_layout()
 
ejercicio_11()

print("\n========== EJERCICIO 13 ==========")
 
def ejercicio_13():
    g = 9.80665
 
    def theta(t):
        return (np.pi / 12) * np.cos(np.pi * t)
 
    def dtheta_dt(t):
        return -(np.pi**2 / 12) * np.sin(np.pi * t)
 
    def d2theta_dt2(t):
        return -(np.pi**3 / 12) * np.cos(np.pi * t)
 
    def sistema(t, estado):
        r, rdot = estado
        th = theta(t)
        dth = dtheta_dt(t)
        d2th = d2theta_dt2(t)
 
        # Ecuación del deslizador en coordenadas polares (barra rígida):
        # r'' - r*θ'² = -g*sin(θ)
        # → r'' = r*θ'² - g*sin(θ)
        dr = rdot
        dr2 = r * dth**2 - g * np.sin(th)
        return [dr, dr2]
 
    h = 0.001
    t = 0.0
    estado = np.array([0.75, 0.0])  # r(0) = 0.75 m, r'(0) = 0
 
    ts = []
    rs = []
 
    t_salida = None
 
    while t <= 5.0:
        ts.append(t)
        rs.append(estado[0])
 
        estado = rk4(sistema, t, estado, h)
        t += h
 
        # Detectar cuando el deslizador llega al extremo de la barra (longitud típica ~ 2 m)
        if estado[0] >= 2.0 and t_salida is None:
            t_salida = t
 
    if t_salida:
        print(f"El deslizador alcanza el extremo de la barra en t ≈ {t_salida:.4f} s")
    else:
        print("El deslizador no alcanzó el extremo en el intervalo simulado.")
        print(f"Posición final: r({ts[-1]:.2f}) = {rs[-1]:.4f} m")
 
    plt.figure()
    plt.plot(ts, rs, color='darkorange')
    plt.axhline(y=2.0, color='red', linestyle='--', label='Extremo barra (2 m)')
    if t_salida:
        plt.axvline(x=t_salida, color='green', linestyle=':', label=f't salida ≈ {t_salida:.3f} s')
    plt.title("Ejercicio 13 — Deslizador sobre barra giratoria")
    plt.xlabel("t (s)")
    plt.ylabel("r (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
 
ejercicio_13()

def ejercicio_15():

    k = 3000
    m = 6
    mu = 0.5
    g = 9.80665

    def sistema(t, Y):

        y = Y[0]
        v = Y[1]

        friccion = -mu*m*g*np.sign(v)

        dy = v
        dv = (-k*y + friccion)/m

        return np.array([dy, dv])

    h = 0.001
    t = 0

    Y = np.array([0.1, 0.0])

    ts = []
    ys = []

    while t <= 0.1:

        ts.append(t)
        ys.append(Y[0])

        Y = rk4(sistema, t, Y, h)
        t += h

    plt.figure()
    plt.plot(ts, ys)
    plt.title("Fricción seca")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid()

ejercicio_15()

print("\n========== EJERCICIO 17 ==========")

def ejercicio_17():

    print("\n--- RESULTADOS EJERCICIO 17 EN GRAFICA ---")

    def sistema(t, Y):

        y = Y[0]
        v = Y[1]

        dy = v
        dv = -0.5*(y**2 - 1)*v - y

        return np.array([dy, dv])

    h = 0.01
    t = 0

    Y = np.array([1.0, 0.0])

    ts = []
    ys = []

    while t <= 20:

        ts.append(t)
        ys.append(Y[0])

        Y = rk4(sistema, t, Y, h)
        t += h

    plt.figure()
    plt.plot(ts, ys)
    plt.title("Van der Pol")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

ejercicio_17()

print("\n========== EJERCICIO 19 ==========")

def ejercicio_19():

    print("\n--- RESULTADOS EJERCICIO 19 EN GRAFICA ---")

    def sistema(x, Y):

        y = Y[0]
        v = Y[1]

        dy = v
        dv = -(v/x) - y

        return np.array([dy, dv])

    h = 0.01

    x = 1e-2

    Y = np.array([1.0, 0.0])

    xs = []
    ys = []

    while x <= 20:

        xs.append(x)
        ys.append(Y[0])

        Y = rk4(sistema, x, Y, h)
        x += h

    plt.figure()
    plt.plot(xs, ys)
    plt.title("Ecuación de Bessel")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

ejercicio_19()

print("\n========== EJERCICIO 21 ==========")
 
def ejercicio_21():
    L = 1.0    # Henrys
    R = 1.0    # Ohms
    C = 1.0    # Farads
 
    def E(t):
        return 100 * np.sin(60 * np.pi * t)
 
    # Estado: [i1, i2, q2]
    def sistema(t, estado):
        i1, i2, q2 = estado
 
        di1 = (E(t) - 3*R*i1 - 2*R*i2) / L
        di2 = (E(t) - q2/C - 2*R*i1 - 3*R*i2) / R   # aproximación sin inductancia en rama 2
        dq2 = i2
 
        return [di1, di2, dq2]
 
    h = 0.0001
    t = 0.0
    estado = np.array([0.0, 0.0, 0.0])
 
    ts = []
    i1s = []
    i2s = []
 
    print(f"\n{'t (s)':<10} {'i1 (A)':<15} {'i2 (A)':<15} {'q2 (C)':<15}")
    print("-" * 55)
 
    pasos_imprimir = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    siguiente_impresion = 0
 
    while t <= 0.1 + h/2:
        ts.append(t)
        i1s.append(estado[0])
        i2s.append(estado[1])
 
        # Imprimir en pasos seleccionados
        if siguiente_impresion < len(pasos_imprimir) and t >= pasos_imprimir[siguiente_impresion] - h/2:
            print(f"{t:<10.4f} {estado[0]:<15.6f} {estado[1]:<15.6f} {estado[2]:<15.6f}")
            siguiente_impresion += 1
 
        estado = rk4(sistema, t, estado, h)
        t += h
 
    plt.figure()
    plt.plot(ts, i1s, label='i₁(t)', color='navy')
    plt.plot(ts, i2s, label='i₂(t)', color='crimson', linestyle='--')
    plt.title("Ejercicio 21 — Circuito eléctrico (Kirchhoff)")
    plt.xlabel("t (s)")
    plt.ylabel("Corriente (A)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
 
ejercicio_21()

plt.show()