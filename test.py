import numpy as np
im = [[[1, 1, 1, 0], [2, 1, 1, 0]],[[1, 2, 1, 0], [2, 2, 1, 0]],[[1, 3, 1, 0],[2, 3, 1, 0]]]
im = np.array(im)
print(im[:,:,0:3])