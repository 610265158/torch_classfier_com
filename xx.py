import editdistance
import numpy as np
xx=editdistance.distance(np.array([0,2,2,3]), np.array([0,1,2]))
print(xx)