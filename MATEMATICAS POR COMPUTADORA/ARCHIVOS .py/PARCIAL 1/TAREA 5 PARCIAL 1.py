import math 
import numpy as np # type: ignore
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve # type: ignore

#=================================================================================================================
#Ejercicio 2.51
"""
Sea f(x) una funcion continua sobre el intervalo [a,b] tal que f(a)f(b)<0. 
Proporciona claramente todos los detalles matemáticos de cómo el El método de la bisección
se aproxima la raíz de f(x) en el intervalo [a,b].
"""

def biseccion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        print("El método de la bisección no se puede aplicar.")
        return None, 0
    
    print(f"{'Iteración':<12} | {'a':>12} | {'b':>12} | {'c':>12} | {'f(c)':>12}")
    print("-"*65)
    
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        
        print(f"{i:<12} | {a:>12.6f} | {b:>12.6f} | {c:>12.6f} | {fc:>12.6f}")
        
        if abs(fc) < tol:
            print(f"\nRaíz aproximada: {c:.6f}")
            print(f"Iteraciones: {i}")
            return c, i
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2, max_iter


# LLAMADA A LA FUNCIÓN
f = lambda x: x**3 - x - 1
biseccion(f, 1, 2, 1e-6, 100)

#==============================================================================================================
#Ejercicio 2.52
"""
Dejar que f(x) sea una función continua  en el intervalo [a,b] donde f(a)*f(b) < 0. 
Proporcione claramente todos los detalles matemáticos de como el método de la regla falsa se aproxima
a la raíz de la función f(x) en el intervalo [a,b].
"""

def regla_falsa(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        print("El método de la regla falsa no se puede aplicar.")
        return None, 0
    
    print(f"{'Iteración':<12} | {'a':>12} | {'b':>12} | {'c':>12} | {'f(c)':>12}")
    print("-"*65)
    
    for i in range(1, max_iter + 1):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        fc = f(c)
        
        print(f"{i:<12} | {a:>12.6f} | {b:>12.6f} | {c:>12.6f} | {fc:>12.6f}")
        
        if abs(fc) < tol:
            print(f"\nRaíz aproximada: {c:.6f}")
            print(f"Iteraciones: {i}")
            return c, i
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    return (a * f(b) - b * f(a)) / (f(b) - f(a)), max_iter


# LLAMADA
f = lambda x: x**3 - x - 1
regla_falsa(f, 1, 2, 1e-6, 100)

#=========================================================================================================================
#Ejercicio 2.53
"""
Dejar que f(x) sea continua con raíz cerca x=x0. 
Claramente proporcione todos los detalles matemáticos de como funciona el método de newton
se aproxima a la raíz en la función f(x).
"""

def newton(f, df, x0, tol, max_iter):
    print(f"{'Iteración':<12} | {'x_n':>12}")
    print("-"*27)
    
    for i in range(1, max_iter + 1):
        if df(x0) == 0:
            print("Error: La derivada es cero.")
            return None, 0
        
        x = x0 - f(x0) / df(x0)
        print(f"{i:<12} | {x:>12.6f}")
        
        if abs(x - x0) < tol:
            print(f"\nRaíz aproximada: {x:.6f}")
            print(f"Iteraciones: {i}")
            return x, i
        
        x0 = x
    
    return x, max_iter


# LLAMADA
f = lambda x: x**3 - x - 1
df = lambda x: 3*x**2 - 1

newton(f, df, 1, 1e-6, 100)

#=========================================================================================================================
#Ejercicio 2.54
"""
Dejar que f(x) sea continua con raíz cerca x=x0.
Claramente proporcione todos los detalles matemáticos de como funciona el método de la secante
se aproxima a la raíz en la función f(x).
"""

def secante(f, x0, x1, tol, max_iter):
    print(f"{'Iteración':<12} | {'x_n':>12}")
    print("-"*27)
    
    for i in range(1, max_iter + 1):
        if f(x1) - f(x0) == 0:
            print("Error: La función no cambia entre x0 y x1.")
            return None, 0
        
        x = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        print(f"{i:<12} | {x:>12.6f}")
        
        if abs(x - x1) < tol:
            print(f"\nRaíz aproximada: {x:.6f}")
            print(f"Iteraciones: {i}")
            return x, i
        
        x0, x1 = x1, x
    
    return x, max_iter


# LLAMADA
f = lambda x: x**3 - x - 1
secante(f, 1, 2, 1e-6, 100)

#=========================================================================================================================
#Ejercicio 2.55
"""
Cuantas iteraciones del método de la bisección son necesarias para aproximar a raíz de 3 dentro de 10 a la menos 3, 
10 a la menos 4 y 10 a la menos 15 usando el intervalo inicial [a,b]= [0,2]. Usar el teorema:

n=log sub 2(|b-a|/epsilon)-1
"""
def iteraciones_biseccion(a, b, epsilon):
    n = math.ceil(math.log2((b - a) / epsilon))
    return n

a, b = 0, 2
epsilons = [1e-3, 1e-4, 1e-15]

for epsilon in epsilons:
    n = iteraciones_biseccion(a, b, epsilon)
    print(f"Para epsilon={epsilon}, se necesitan {n} iteraciones.")

#=========================================================================================================================
#Ejercicio 2.56
"""
Vuelve a consultar el ejemplo 2.1 y demuestra que tienes los mismos resultados resolviendo el problema x cubica - 3 = 0.
Genere versiones versiones de todos los gráficos de forma del ejemplo y proporcione descripciones detalladas de lo que
aprende de cada gráfico. 
"""
def f(x):
    return x**3 - 3

# Raíz exacta
alpha = 3**(1/3)

# Método de bisección guardando errores
def biseccion_error(f, a, b, n_iter):
    errores = []
    iteraciones = []
    
    for k in range(n_iter):
        c = (a + b)/2
        error = abs(c - alpha)
        
        errores.append(error)
        iteraciones.append(k)
        
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
            
    return np.array(iteraciones), np.array(errores)

# Ejecutamos 30 iteraciones
iters, errores = biseccion_error(f, 1, 2, 30)

# ---------------------------
# Gráfico 1: Error vs Iteración
# ---------------------------
plt.figure()
plt.scatter(iters, errores)
plt.xlabel("Número de Iteración")
plt.ylabel("Error Absoluto")
plt.title("Bisection Method Error vs Iteration (x^3 - 3)")
plt.show()

# ---------------------------
# Gráfico 2: log2(Error) vs Iteración
# ---------------------------
plt.figure()
plt.scatter(iters, np.log2(errores))
plt.xlabel("Número de Iteración")
plt.ylabel("Base 2 Log of Absolute Error")
plt.title("Bisection Method log2(Error) vs Iteration")
plt.show()

# ---------------------------
# Gráfico 3: Error_k+1 vs Error_k
# ---------------------------
plt.figure()
plt.scatter(errores[:-1], errores[1:])
plt.xlabel("Error en iteración k")
plt.ylabel("Error en iteración k+1")
plt.title("Error_{k+1} vs Error_k")
plt.show()

#=========================================================================================================================
#Ejercicio 2.57
"""
En este problema demostraras que todos tus códigos para encontrar la raíz. En el comienzo del capitulo
 se propusieron la ecuación 
para resolver el problema.

3sinx+9x=x^2-cosx

Escribe un script que llama al método de Newton, Secante, Regla Falsa y Bisección para encontrar la solución positiva
de la ecuación. El script debe generar la solución de una forma clara y legible.
"""

def f(x):
    return 3*math.sin(x) + 9*x - x**2 + math.cos(x)

def df(x):
    return 3*math.cos(x) + 9 - 2*x - math.sin(x)

# =====================================================
# MÉTODO DE BISECCIÓN
# =====================================================

def biseccion(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, 0
    
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c, i
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return c, max_iter

# =====================================================
# MÉTODO DE REGLA FALSA
# =====================================================

def regla_falsa(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, 0
    
    for i in range(1, max_iter + 1):
        c = (a*f(b) - b*f(a)) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c, i
        
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    
    return c, max_iter

# =====================================================
# MÉTODO DE NEWTON
# =====================================================

def newton(f, df, x0, tol, max_iter):
    for i in range(1, max_iter + 1):
        if df(x0) == 0:
            return None, 0
        
        x = x0 - f(x0)/df(x0)
        
        if abs(x - x0) < tol:
            return x, i
        
        x0 = x
    
    return x, max_iter

# =====================================================
# MÉTODO DE LA SECANTE
# =====================================================

def secante(f, x0, x1, tol, max_iter):
    for i in range(1, max_iter + 1):
        if f(x1) - f(x0) == 0:
            return None, 0
        
        x = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        
        if abs(x - x1) < tol:
            return x, i
        
        x0, x1 = x0, x
    
    return x, max_iter

# =====================================================
# SCRIPT PRINCIPAL
# =====================================================

tol = 1e-6
max_iter = 100

print("\nResolviendo: 3sin(x)+9x = x^2 - cos(x)\n")

# Elegimos intervalo donde cambia signo
a, b = 0, 2

# Bisección
rb, ib = biseccion(f, a, b, tol, max_iter)

# Regla Falsa
rr, ir = regla_falsa(f, a, b, tol, max_iter)

# Newton
rn, inew = newton(f, df, 1.5, tol, max_iter)

# Secante
rs, isec = secante(f, 1, 2, tol, max_iter)

# =====================================================
# RESULTADOS
# =====================================================

print("Método           | Raíz aproximada | Iteraciones")
print("------------------------------------------------")

if rb is not None:
    print(f"Bisección        | {rb:.6f}        | {ib}")
else:
    print("Bisección        | No converge     | 0")

if rr is not None:
    print(f"Regla Falsa      | {rr:.6f}        | {ir}")
else:
    print("Regla Falsa      | No converge     | 0")

if rn is not None:
    print(f"Newton           | {rn:.6f}        | {inew}")
else:
    print("Newton           | No converge     | 0")

if rs is not None:
    print(f"Secante          | {rs:.6f}        | {isec}")
else:
    print("Secante          | No converge     | 0")

#=========================================================================================================================
#Ejercicio 2.58
"""
Un método de búsqueda de las raíces tiene una tasa de convergencia de orden M si hay constante C tal que:

|Xsubk - Xsubk+1| <= C * |Xsubk - Xsubk+1|**M

Aquí, x es la raíz exacta , donde xsubK es la K a la th iteración de la raíz técnica de la búsqueda,
y x sub K+1 es el (K+1) a la st iteración de la técnica de búsqueda de raíces. 

a. Si se considera la ecuación anterior y tomamos el logaritmo en base 10 de ambos lados entonces obtenemos:
registro(|Xsubk - Xsubk+1|)<__________________________+_____________________________

b. En la parte de (a) debería haberse descubierto que el logaritmo del nuevo error es un función lineal del
logaritmo del error anterior. ¿Cuál es la pendiente de está función lineal en un gráfico?

c. A continuación se verán 6 gráficos lineales diferentes del nuevo error al error anterior para diferentes
técnicas para la búsqueda de raíces.
¿Cuál es el orden de la tasa de convergencia aproximada para cada uno de los métodos?

d. Bajo tus propias palabras responde:

    a) ¿Que significa que un método de la búsqueda aproximada de la raíz tenga una tasa de convergencia de primer orden?

    b) ¿Y con una tasa de convergencia de segundo orden?
"""
#==========================INCISO A====================================

print("RESPUESTA INCISO (a)\n")

print("Partimos de la definición del orden de convergencia:")
print("    e_{k+1} <= C * (e_k)^M\n")

print("Tomando logaritmo base 10 en ambos lados:")
print("    log10(e_{k+1}) <= log10(C * (e_k)^M)\n")

print("Usando propiedades de logaritmos:")
print("    log10(ab) = log10(a) + log10(b)")
print("    log10(a^M) = M log10(a)\n")

print("Obtenemos:")
print("    log10(e_{k+1}) <= log10(C) + M log10(e_k)\n")

#=======================INCISO B=====================================

print("--------------------------------------------------")
print("RESPUESTA INCISO (b)\n")

print("La ecuación obtenida tiene la forma:")
print("    y = Mx + log10(C)\n")

print("Si definimos:")
print("    y = log10(e_{k+1})")
print("    x = log10(e_k)\n")

print("Entonces la pendiente de la recta es:")
print("    M\n")

print("Conclusión:")
print("La pendiente del gráfico log(error nuevo) vs log(error anterior)")
print("es exactamente el orden de convergencia M.")

#=====================INCISO D=====================================================
print("--------------------------------------------------")
print("RESPUESTAS PUNTO (d)\n")

print("a) Convergencia de primer orden:")
print("Un método tiene convergencia de primer orden cuando el error en la siguiente")
print("iteración es proporcional al error actual:")
print("    e_{k+1} ≈ C * e_k")
print("Esto significa que el error disminuye a un ritmo constante.")
print("El número de cifras correctas aumenta de manera lineal.")
print("Ejemplo: método de Bisección o Regla Falsa.\n")

print("b) Convergencia de segundo orden:")
print("Un método tiene convergencia de segundo orden cuando el error satisface:")
print("    e_{k+1} ≈ C * e_k^2")
print("Esto significa que el error se eleva al cuadrado en cada iteración.")
print("El número de cifras correctas aproximadamente se duplica en cada paso.")
print("La convergencia es mucho más rápida.")
print("Ejemplo: método de Newton.")

#====================================================================================================================
#Ejercicio 2.59

"""
Un alumno empezó a utilizar el método de Newton para resolver un problema de la búsqueda de raíces. 
Para probar su código se uso una ecuación cuya solución se conocía. Dado el punto de partida, el error absoluto después
de uno de los pasos del método de Newton fue |x1-x*|=0,2, ¿Cual es el error esperado en el paso 2?
¿Que pasa en el paso 3 y 4?

Defiende tus respuestas describiendo completamente tu proceso de pensamiento.
"""
print("\nEJERCICIO 2.59\n")

error = 0.2

print("Iteración | Error aproximado")
print("----------------------------")

for k in range(1, 5):
    print(f"{k:<9} | {error:.10f}")
    error = error**2

#=======================================================================================================================
#Ejercicio 2.60
"""
Hay varias formas de obtener las raíces de las que se estuvieron viendo hasta ahora.
Se pueden construir estos métodos utilizando las series de Taylor de esta manera

Cerca x=x0 la función f(x) se aproxima mediante Taylor

f(x) se aproxima a y  = f(xsubcero)+la sumatoria de n =1 donde ((f^n)(xsubcero)/n!)(x-xsubcero))

Donde N es un entero positivo. En el algoritmo de la búsqueda de raíces y a cero para encontrar la raíz de la función de aproximación.
La raíz de esta función debería estar cerca de la raíz real que se está buscando. Para resolver la siguiente 
iteración se usa la ecuación:

0=f(xsubcero)+la sumatoria de n =1 donde ((f^n)(xsubcero)/n!)(x-xsubcero))

    a) Resolver para x en el caso que N=2 . Luego describa la función Python.

    b) Demuestre que su código del inciso a realmente funciona resolviendo varios problemas en el
    que conoce la solución exacta.

    c) Muestra varios gráficos que estiman el orden de método a partir de la parte (a). Cree un gráfico
    log-log de los errores sucesivos para varios problemas diferentes de la resolución de ecuaciones.

    d) ¿Cuáles son los pros y los contras de utilizar este nuevo método?
"""

def taylor_segundo_orden(f, df, ddf, x0):
    A = ddf(x0)/2
    B = df(x0)
    C = f(x0)

    discriminante = B**2 - 4*A*C

    if discriminante < 0:
        return None

    h1 = (-B + math.sqrt(discriminante))/(2*A)
    h2 = (-B - math.sqrt(discriminante))/(2*A)

    # Elegimos el h más pequeño en magnitud
    h = h1 if abs(h1) < abs(h2) else h2

    return x0 + h

#=======================================================================================================================
#Ejercicio 2.61
"""
Un objeto que cae verticalmente por el aire está sujeto a fricción debido a tanto a la resistencia del
aire como a la gravedad. La función que describe la posición de tal función es:

s(t)=s_subcero -(mg/k)t+(m^2*g/k^2)(1-e^(-kt/m))

a) ¿Cuáles son las unidades del parámetro  k?

b) Si m =1, g = 9.81 m/s2 , k = 0.1 y S0= 100 m ¿Cuanto tiempo sera necesario para que el objeto golpeé el suelo?,
Encuentra tu respuesta dentro de 0,001 s

c) El valor de  k depende de la aerodinámica del objeto y podría ser difícil de medir.
Queremos realizar un análisis de sensibilidad sobre su respuesta a la parte (b) sujeta a pequeños errores de medición en  k.
Si el valor de k sólo se conoce dentro del 10 por cietno de lo que son ¿Sus estimaciones de cuándo el objeto tocará el suelo?
"""

def s(t, m=1, g=9.81, k=0.1, s0=100):
    return s0 - (m*g/k)*t + (m**2 * g / k**2)*(1 - math.exp(-k*t/m))

def ds(t, m=1, g=9.81, k=0.1):
    return -(m*g/k) + (m*g/k)*math.exp(-k*t/m)

raiz, it = newton(lambda t: s(t), lambda t: ds(t), 5, 1e-6, 100)

print("Tiempo de impacto:", raiz)

#========================================================================================================================
#Ejercicio 2.62
"""
¿se puede utilizar el método de la bisección, el método de la regla falsa o el método de Newton para encontrar 
las raíces de la función  f(x)=cos(x)+1? ¿Explicar por qué sí o por qué no para cada técnica?
"""

print("Función: f(x) = cos(x) + 1\n")

print("1) Método de la Bisección:")
print("No se puede utilizar porque el método requiere cambio de signo en el intervalo.")
print("La función cos(x) + 1 es siempre mayor o igual a 0.")
print("Por lo tanto, no existe un intervalo donde f(a)*f(b) < 0.\n")

print("2) Método de la Regla Falsa:")
print("Tampoco se puede utilizar porque también requiere cambio de signo.")
print("La función nunca es negativa.\n")

print("3) Método de Newton:")
print("Sí se puede utilizar porque no requiere cambio de signo.")
print("Sin embargo, la derivada en la raíz es cero (f'(π) = 0),")
print("por lo que la convergencia puede ser lenta o inestable.")
print("Con un buen punto inicial cercano a π, el método puede converger.")

#========================================================================================================================
#Ejercicio 2.63

"""
En Cálculo de Variable Única estudiaste métodos para encontrar variables locales y extremos globales de funciones. 
Probablemente recuerdes esa parte del proceso es establecer la primera derivada en cero y resolver la independiente variable
(recuerda por qué estás haciendo esto).
    
El problema con esto es que puede ser muy, muy difícil resolver a mano. Este es un lugar perfecto para el método de
Newton o cualquier otro con técnica de búsqueda de raíces.
    
Encuentre los extremos locales para la función:

f(x)=x^3(x-3)(x-6)^4
"""

# Definir variable simbólica
x = sp.symbols('x')

# Definir función
f = x**3 * (x-3) * (x-6)**4

# Derivadas
f1 = sp.diff(f, x)
f2 = sp.diff(f1, x)

# Convertir a funciones numéricas
f1_num = sp.lambdify(x, f1, "numpy")
f2_num = sp.lambdify(x, f2, "numpy")
f_num = sp.lambdify(x, f, "numpy")

# Método de Newton para f'(x)=0
def newton(df, x0, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        dfx = df(x0)
        if abs(dfx) < tol:
            return x0
        x0 = x0 - df(x0)/f2_num(x0)
    return x0

# Intentar varios puntos iniciales
puntos_iniciales = [-2, 1, 4, 7]
raices = []

for p0 in puntos_iniciales:
    raiz = newton(f1_num, p0)
    raiz = round(raiz, 6)
    if raiz not in raices:
        raices.append(raiz)

print("Puntos críticos encontrados:\n")

for r in raices:
    valor = f_num(r)
    segunda = f2_num(r)
    
    if segunda > 0:
        tipo = "Mínimo local"
    elif segunda < 0:
        tipo = "Máximo local"
    else:
        tipo = "Punto de inflexión"
    
    print(f"x = {r}")
    print(f"f(x) = {valor}")
    print(f"Clasificación: {tipo}\n")

#========================================================================================================================
#Ejercicio 2.64
"""
El punto fijo de una función f(x) es un método que resuelve la ecuación f(x)=x. Los puntos fijos son interesantes 
en los procesos iterativos ya que son puntos fijos que no cambian con la función repetida de f.

Por ejemplo, considere la función  f(x)=x2-6. Los puntos fijos de  f(x) se puede encontrar resolviendo 
la ecuación  x cuadrada -6=x, cuando simplificado algebraica-mente, es  x cuadrada -x-6=0. Factorizando el 
lado izquierdo se obtiene  (x-3)(x+2)=0 lo cual implica que  x=3y  x=-2 son puntos fijos para esta función. 
Así es,  f(3) = 3y f(-2) = -2. 
    
Observe, sin embargo, que aunque el hallazgo se solucionó.
    
Los puntos son idénticos a un problema de búsqueda de raíces.

    a. Utilice un algoritmo numérico de búsqueda de raíces para encontrar los puntos fijos de
    la función  f(x)=x cuadrada -6 en el intervalo  [0,∞).

    b. Encuentra los puntos fijos de la función

    f(x)=raíz de (8/(x+6))
"""

def g(x):
    return x**2 - 6 - x

def dg(x):
    return 2*x - 1

def newton(g, dg, x0, tol=1e-8, max_iter=100):
    for i in range(max_iter):
        x1 = x0 - g(x0)/dg(x0)
        if abs(x1 - x0) < tol:
            return x1, i+1
        x0 = x1
    return x0, max_iter

raiz, it = newton(g, dg, 4)

print("Punto fijo en [0,∞):", round(raiz,6))
print("Iteraciones:", it)

#========================================================================================================================
#Ejercicio 2.65
"""
El spyci, la biblioteca en Python tiene muchas rutinas de análisis numérico integradas muy similares a las que hemos creado en este capítulo.
De particular interés para la tarea de búsqueda de raíces es el fsolve con el comando de la biblioteca spyci.optimize.

Siga las instrucciones del archivo y usando el código resuelva el sistema de ecuaciones no lineales.
"""
# Definimos la función
def f(x):
    return x*np.sin(x) - np.log(x)

# Dominio (evitamos 0 por el logaritmo)
x_vals = np.linspace(0.1, 5, 400)
y_vals = f(x_vals)

plt.figure()
plt.axhline(0)
plt.plot(x_vals, y_vals)
plt.title("f(x) = x sin(x) - ln(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
# Definimos el sistema
def sistema(vars):
    x, y = vars
    eq1 = x**2 - x*y**2 - 2
    eq2 = x*y - 2
    return [eq1, eq2]

# Punto inicial
x0 = 3

# Resolver con diagnósticos
sol, info, ier, msg = fsolve(f, x0, full_output=True)

print("Solución aproximada:", sol[0])
print("\nDiagnóstico del solver:")
print("Código ier =", ier)
print("Mensaje =", msg)
print("Evaluaciones de función =", info['nfev'])

# Punto inicial (muy importante elegirlo bien)
x0 = [1.5, 1]

# Llamamos fsolve con diagnósticos completos
solucion, info, ier, msg = fsolve(sistema, x0, full_output=True)

print("Solución encontrada:")
print("x =", solucion[0])
print("y =", solucion[1])

print("\nDiagnóstico del solver:")
print("Código ier =", ier)
print("Mensaje =", msg)
print("Número de evaluaciones de función =", info['nfev'])