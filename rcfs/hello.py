from ulab import numpy as np
A = np.array([[1,2],[3,4]])
B = np.linalg.inv(A)
C = np.dot(A, B) 
print(C)
