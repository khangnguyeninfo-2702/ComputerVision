import numpy as np


def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
    def cross(x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
    A1 = np.matmul(cross(x1),P1)
    A2 = np.matmul(cross(x2),P2)
    A = np.vstack((A1, A2))
    vals,vecs = np.linalg.eig(np.dot(A.T,A))
    x_homogenous = vecs[:,np.argmin(vals)]
    X = x_homogenous/x_homogenous[3] #Normalize
    ##-your-code-ends-here-##
    
    return X
