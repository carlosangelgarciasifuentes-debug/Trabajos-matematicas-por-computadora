import numpy as np
#Crear la matrix
np.set_printoptions(precision=3, suppress=1, floatmode='fixed')

print("\n")

print("===========ESTE ES UN EJEMPLO DE MATRIZ==================")

print("\n")

A=np.array([[1,2,3,4],[-2,-3,-4,6],[-1,3,-5,-8]])

print("A =")
print(A)

print("\n")

#Obtener el tamaño de A

m,n = np.shape(A)
print("filas = {}, columnas = {}".format(m,n))

print("\n")

#accedemos a la segund fila y tercer columna

a23 = A[1,2]
print("El valor de la segunda fila y tercera columna es: ", a23)

print("\n")

#Encontrar la tercera fila de la matriz

fila3 = A[2,:]
print("La tercera fila de la matriz es: ", fila3)

print("\n")

#Encontrar la segunda columna de la matriz
columna2 = A[:,1]
print("La segunda columna de la matriz es: ", columna2)

print("\n")

'''
B = A 

#Cambiar los valores de B
B[0,0]= 2

print("B=")
print(B)

print("A=")
print(A)

'''

#Para crear una copia de A sin alterarla usando A=B hacemos:

B = A.copy()
B[0,0] = 2

print("B=")
print(B)

print("\n")

print("A=")
print(A)

print("\n")

#operaciones: suma y resta

x=np.array([[1,2],[3,4]])
y=np.array([[-1,3],[2,-5]])

print("X= ")
print(x)

print("\n")

print("Y=")
print(y)

print("\n")

#hacemos la suma de X y Y

suma = x + y
print("La suma de X + Y es:")
print(suma)

print("\n")

# hacemos la resta

resta = x - y
print("La resta de X + Y es:")
print(resta)

#hacemos la multiplicación de X y Y

multiplicacion = x * y
print("La suma de X por Y es:")
print(multiplicacion)

print("\n")

#hacemos la división de X y Y

division = x / y
print("La división de X y Y es:")
print(division)

print("\n")

#multiplicación escalr con matriz

escalar = 10
multiescalar = escalar * x

print("La multiplicación de X con el escalar es: ")
print(multiescalar)

print("\n")

#producto punto de x con y
A= np.array([[1,2,3],[-1,-2,-3]])
B= np.array([[1,2],[0,4],[-3,2]])
prod_punto = np.dot(A,B)

print("A =")
print(A)

print("\n")

print("B =")
print(B)

print("\n")

print("El producto punto de A con B es: ")
print(prod_punto)

print("\n")

#Multiplicando un vector A por la matriz identidad

I3= np.array([[1,0,0],[0,1,0],[0,0,1]])
I2= np.array([[1,0],[0,1]])

C= np.dot(A, I3)
print("El resultado de multiplicar la matriz A por la matriz identidad I3 es: ")
print(C)

print("\n")

D = np.dot(I2, A)
print("El resultado de multiplicar la matriz identidad I2 por la matriz A es: ")
print(D)

print("\n")

#Calculando la inversa de una matriz
MAT= np.array([[1,2,3],[-1,2,-3],[0,2,5]])

print("La matriz MAT es: ")
print(MAT)

print("\n")

MAT_INV = np.linalg.inv(MAT)
print("La inversa de la matriz MAT es: ")
print(MAT_INV)

print("\n")

#Verificamos que la inversa sea correcta iniviertiendola para regresar a la matriz

MAT2_INV = np.linalg.inv(MAT_INV)

print("La inversa de la inversa de MAT es: ")
print(MAT2_INV)

print("\n")

#Multiplicando la inversa con la matriz original para obtener la identidad
IDENTIDAD = np.dot(MAT, MAT_INV)
print("El resultado de multiplicar la matriz MAT por su inversa es: ")
print(IDENTIDAD)

print("\n")
