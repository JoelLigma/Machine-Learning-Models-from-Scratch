"""
Linear algebra exercise.

1. Computing the determinant of a matrix from scratch
2. Computing the approximate eigenvector of a square matrix from scratch
3. Computing the approximate dominant eigenvalue from scratch

"""
import numpy as np


def create_matrix_copy(original_matrix):
    matrix_copy = []
    for i in range(len(original_matrix)):
        temp_row = []
        for j in range(len(original_matrix)):
            temp_row += [original_matrix[i][j]]
        matrix_copy += [temp_row]
    return matrix_copy

def compute_determinant(matrix):
  """
  Returns the determinant (scalar value) of a square matrix 
  -------
  
  Implementation details:
  If the n x n matrix has n > 2, we need to rely on the cofactor expansion along the first row
  we can write a function as follow:
  - use recursion to find the determinant of smaller matrices which we obtain by deleting 
    the appropriate rows and columns
  - we stop the recusion once the base case of the 2 x 2 matrix is reached as we can find 
    the determinant of that square matrix

  1. omit first row and col and compute the determinant of the remaining square matrix (in the 3x3 case)
  2. Next, omit the first row and second column and do the same again etc.
  We break down the problem to the base case and solve it from there.
  """
  
    # start by checking the number of rows
    n_rows = len(matrix)

    for row in matrix:
        # check if the input is a square matrix
        if len(row) != n_rows:
            print("The input matrix is not a square matrix.")
            # cannot compute the determinant so return none
            return 

        # next check for the base case which is a 2x2 matrix 
        if len(matrix) == 2:
            # from here we can simply compute the determinant
            determinant = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
            return determinant

        # However, if the base case condition is not met and we encounter a larger matrix,
        # we use the Laplace cofactor expansion recursively until we reach the base case
        else:
            output = 0
            n_cols = n_rows
            for j in range(n_cols):

                # create copy of matrix
                matrix_copy = create_matrix_copy(matrix)

                # remove first row 
                matrix_copy = matrix_copy[1:]
                
                smaller_matrix_lst = []
                for k in range(len(matrix_copy)):
                    smaller_matrix_lst += [matrix_copy[k]]
                    # remove column at column index j 
                    smaller_matrix_lst[k].pop(j)              
                # recursively call the compute_deterimant function
                cofactor = (-1) ** j * matrix[0][j] * compute_determinant(smaller_matrix_lst)
                output += cofactor
            return output


def compute_eigenvector(A, n_iter=50):
    """
    Power iteration algorithm without numpy.
    """
    import random

    # get random vector v which we will use as a start to approximate the dominant eigenvalue
    v = [random.uniform(0,1) for _ in range(len(A))]
       
    # get all eigenvalues
    for _ in range(n_iter):
       # calculate the dot product between matrix A and vector v
        v_1 = [sum([row[i] * v[i] for i in range(len(v))]) for row in A]

        # calculate the L2-norm of v_1
        v_1_L2_norm = sum([v**2 for v in v_1]) ** (1/2)
        
        # re-normalize the vector
        v = [v_1[i] / v_1_L2_norm for i in range(len(v_1))]
        
    return v  


def rayleigh_quotient(A, v):
    """
    Returns the Rayleigh quotient (approximate eigenvalue of a given eigenvector) 
    of a given hermitian matrix A and vector v.

    Formula: lambda = v.T * A*v / v.T * v

    Note: A check whether the input matrix is hermitian may be necessary to implement.
    """

    # the dot product of v_transpose and matrix A
    vT_dot_A = [sum([row[i] * v[i] for i in range(len(v))]) for row in A]

    # the dot product of the previous result and vector v
    vT_dot_A_dot_v = sum([vT_dot_A[i] * v[i] for i in range(len(v))])

    # the dot product of vector v and vector v
    v_dot_v = sum([v[i] * v[i] for i in range(len(v))])

    return round(vT_dot_A_dot_v / v_dot_v)


def hotellings_deflation(A, v, lamda):
    """
    After finding the approximation of the eigenvector and dominant eigenvalue lambda,
    we compute matrix B which behaves like matrix A for anything orthogonal for vector v
    and thereby zeros out vector v.

    Thus, as the power method + rayleigh quotient will give us the approximation of the largest 
    eigenvalue, we can replace the largest one with zero and retrieve the next largest one and so on.
   
    Formula: A - lambda * v*v.T
    """
    import math
    # dot product of vector v and vector v_transpose
    v_dot_vT = [v[i]*v[i] for i in range(len(v))][0] # take first element
    
    # compute lambda * v*v.T
    right_side = lamda * v_dot_vT

    # subtract the scalar from each element in the matrix A
    B = []
    for row in range(len(A)):
        temp_row = []
        for col in range(len(A)):
            temp_row += [round(A[row][col] - right_side) if A[row][col] - right_side > 0.1*math.exp(-5) else abs(A[row][col] - right_side)]
        B += [temp_row]
    return B


if __name__ == "__main__":
  A = np.array([[7,4,1],
                [4,4,4],
                [1,4,7]])
  
  lambdas = []
  for i in range(len(A)):

      # compute the eigenvector 
      v = compute_eigenvector(A, n_iter=50)

      # retrieve the eigenvalue
      lamda = rayleigh_quotient(A, v)

      # keep track of the results
      lambdas += [lamda]

      # deflate the matrix A
      A = hotellings_deflation(A, v, lamda)

  print("My result:", lambdas)
  print("Expected result: 12, 6, 0")
  print("Determinant:", round(compute_determinant(A)))


"""
Learning takeaway:

- a deeper understanding of the topics invovled.
- after the first two iterations, we encounter a problem. It seems that this may have to 
  do with numeric instability. Hence, numpy is commonly used as the implementations in this
  package are more robust to this issue.
"""
