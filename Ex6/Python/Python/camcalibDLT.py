import numpy as np


def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinatesm with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """

    # Create the matrix A 
    ##-your-code-starts-here-##
    world_coordinates_number = x_world.shape[0]
    A =[]
    for i in range(world_coordinates_number):
        Xn = x_world[i]
        xn, yn, wn = x_im[i]
        # Normalization
        u = xn / wn
        v = yn / wn
        # Row 1:[ 0^T, X_n^T,-y_n*X_n^T ]
        #Row 2:[X_n^T, 0^T,-x_n*X_n^T ]
        A.append(np.hstack((np.zeros(4), Xn, -v * Xn)))
        A.append(np.hstack((Xn, np.zeros(4), -u * Xn)))

    A = np.array(A)
    ##-your-code-ends-here-##

    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    ATA = np.dot(A.T, A)
    eigenvalues,eigenvectors = np.linalg.eig(ATA)
    smallest_ev_idx = np.argmin(eigenvalues)
    ev = eigenvectors[:, smallest_ev_idx]
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(ev, (3, 4))  # here ev is the eigenvector from above
    
    return P
