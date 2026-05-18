import numpy as np
np.set_printoptions(suppress=True, floatmode='fixed')

#Creamos la matriz
A=np.array([[1,1,1],[2,-3,1],[2,3,1],[1,2,-3]])
B=np.array([[6],[-1],[-4]])

Ainv=np.linalg.inv(A)
X=np.dot(Ainv,A)
         
print("La solución de mi sistema es: ")
print(X)