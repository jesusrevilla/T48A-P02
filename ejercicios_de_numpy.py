"""Ejercicios de numpy.ipynb
##CUEVAS GONZALEZ ARELI ALEJANDRA 175454 
## Sigue las instrucciones para cada celda y al final baja la libreta en formato .py y subelo al repositorio que se te indique, no modifiques el nombre la de función, solo importa una vez numpy en el código.
"""
import numpy as np

 '''1. Crea un arreglo de numpy con 10 enteros aleatorios entre 0 y 100.'''
def rand_int():
  return np.random.randint(0, 100, 10)
np.random.seed(10)
rand_int()
"""2. Create a NumPy array of 5 random floating-point numbers between 0 and 1."""
def rand_float():
    return np.random.rand(5)
  np.random.seed(10)
rand_float().__repr__()
"""3. Create a NumPy array of the first 10 prime numbers."""
def first_10_primes():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    return np.array(primes)
first_10_primes()
"""4. Create a NumPy array of the squares of the numbers from 1 to 10."""
  '''Regresa un arreglo de numpy con los cuadrados de los números del 1 al 10.
  '''
def squares():
    return np.arange(1, 11) ** 2
squares()
"""5. Create a NumPy array of the cubes of the numbers from 1 to 10."""
  '''Regresa un arreglo de numpy con los cubos de los números del 1 al 10.
  '''
def cubes():
    return np.arange(1, 11) ** 3

cubes()
"""### NumPy Array Operations

1. Add two NumPy arrays together.
"""
  '''Regresa la suma de dos arreglos de numpy.'''
def add_arrays(arr1, arr2):
  assert arr1.shape == arr2.shape, "Arrays must have the same shape"
  return arr1 + arr2
add_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
add_arrays(arr1, arr2)
"""2. Subtract two NumPy arrays from each other, second argument less first."""

def subtract_arrays(arr1, arr2):
    assert arr1.shape == arr2.shape, "Arrays must have the same shape"
    return arr2 - arr1

subtract_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
subtract_arrays(arr1, arr2)
  '''Calcula arr2 menos arr1 (arreglos de numpy).
  '''

"""3. Multiply two NumPy arrays together (element-wise)."""

def multiply_arrays(arr1, arr2):
  '''Multiplica dos arreglos de numpy elemento por elemento.
  '''
    assert arr1.shape == arr2.shape, "Los arreglos deben tener el mismo tamaño"
    return arr1 * arr2
multiply_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))


arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
multiply_arrays(arr1, arr2)
"""4. Divide two NumPy arrays by each other (element-wise)."""

def divide_arrays(arr1, arr2):
    '''Divide arr2 antre arr1 (arreglos de numpy).

    Precondition
    ------------
      - arr2.any(0)
    '''
    assert arr1.shape == arr2.shape, "Arrays must have the same shape"
    assert not np.any(arr1 == 0), "No se puede dividir por cero"
    return arr2 / arr1
divide_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
divide_arrays(arr1, arr2)
"""5. Create a NumPy array of the integer numbers from 1 to 5. Calculate the mean, median, and standard deviation."""

def stats(arr):
    assert arr.size == 5, 

  mean = np.mean(arr)
  median = np.median(arr)
  std_dev = np.std(arr)

  return (mean, median, std_dev)

arr = np.array([1, 2, 3, 4, 5])
stats(arr)
  '''Calcula la media, la mediana y la desviación estándar de un arreglo de numpy
  en un tuple con las siguientes posiciones: (media, mediana, desviacion_std).

  Returns
  -------
  tuple
    Tuple con las siguientes posiciones: (media, mediana, desviacion_std).

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de los números de 1 a 5.

  Precondition
  ------------
    - arr.size == 5
  '''
  

"""### NumPy Array Indexing and Slicing

1. Create a NumPy array of 10 random integers between 0 and 100. Select the first 5 elements of the array.
"""

def first_5(arr):
    assert arr.size == 10, 
  return arr[:5]

random_array = np.random.randint(0, 100, 10)
print(random_array)
first_5(random_array)
  '''Regresa los primeros 5 elementos de un arr (arreglo) que contiene 10 números
  aleatoreos enteros entre 0 y 100.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 10 elementos con numeros aleatorios del 1 al 100.

  Precondition
  ------------
    - arr.size == 10
  '''
  

"""2. Create a NumPy array of 10 random integers between 0 and 100. Select the last 3 elements of the array."""

def last_3(arr):
    assert arr.size == 10, 
  return arr[-3:]

random_array = np.random.randint(0, 100, 10)
print(random_array)
last_3(random_array)
  '''Regresa los últimos 3 elementos de un arr (arreglo) de numpy que contiene 10
  números enteros aleatoreos entre 0 y 100.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 10 elementos con numeros aleatorios del 1 al 100.

  Precondition
  ------------
    - arr.size == 10
  '''
  

"""3. Create a NumPy array of 10 random integers between 0 and 100. Select the elements at indices 2, 4, and 6."""

def indices_2_4_6(arr):
    assert arr.size == 10, 
  return arr[[2, 4, 6]]

random_array = np.random.randint(0, 100, 10)
print(random_array)
indices_2_4_6(random_array)
  '''Regresa los elementos en los índices 2, 4 y 6 de un arr (arreglo) que contiene
  10 números enteros aleatoreos entre 0 y 100.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 10 elementos con numeros aleatorios del 1 al 100.

  Precondition
  ------------
    - arr.size == 10
  '''
  

"""4. Create a NumPy array of 10 random integers between 0 and 100. Select the elements with values greater than 50."""

def greater_50(arr):
     assert arr.size == 10, 
  return arr[arr > 50]

random_array = np.random.randint(0, 100, 10)
print(random_array)
greater_50(random_array)
  '''Regresa los elementos del arr (arreglo) que contiene 10 números enteros
  aleatoreos entre 0 y 100 que son mayores a 50.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 10 elementos con numeros aleatorios del 1 al 100.

  Precondition
  ------------
    - arr.size == 10
  '''
 

"""5. Create a NumPy array of 10 random integers between 0 and 10. Select elements less than or equal to 7."""

def less_7(arr):
    assert arr.size == 10, 
  return arr[arr <= 7]

random_array = np.random.randint(0, 100, 10)
print(random_array)
less_7(random_array)
  '''Regresa los elementos del arr (arreglo) que contiene 10 números enteros
  aleatoreos entre 0 y 100 que son menores o iguales a 7.

  Parameters
  ----------
  arr: numpy.ndarray
    - arr: arreglo de numpy de 10 elementos con numeros aleatorios del 1 al 100.

  Precondition
  ------------
    - arr.size == 10
  '''
  

"""### NumPy Array Reshaping

1. Create a NumPy array of 12 numbers. Reshape the array into a 2x6 matrix.
"""

def reshape_2x6(arr):
    assert arr.size == 12, 
  return arr.reshape(2, 6)

my_array = np.arange(12)
reshape_2x6(my_array)
  '''Regresa un arreglo de numpy con 12 números y lo convierte en un arreglo de 2x6.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 12 elementos.

  Precondition
  ------------
    - arr.size == 12
  '''
  

"""2. Create a NumPy array of 24 numbers. Reshape the array into a 2x3x4 tensor."""

def reshape_2x3x4(arr):
    assert arr.size == 24, 
  return arr.reshape(2, 3, 4)

my_array = np.arange(24)
reshape_2x3x4(my_array)
  '''Conviert un arreglo de numpy con 24 números en un arreglo de 2x3x4.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 24 elementos.

  Precondition
  ------------
    - arr.size == 24
  '''
  

"""3. Create a NumPy array of 100 numbers. Reshape the array into a 10x10 matrix."""

def reshape_10x10(arr):
    assert arr.size == 100, 
  return arr.reshape(10, 10)

sample_array = np.arange(100)
reshape_10x10(sample_array)
  '''Convierte un numpy array en un numpy array de 10x10.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 100 elementos.

  Precondition
  ------------
    - arr.size == 100
  '''
  

"""4. Create a NumPy array of 1000 numbers. Reshape the array into a 10x10x10 tensor."""

def reshape_10x10x10(arr):
      assert arr.size == 1000, 
  return arr.reshape(10, 10, 10)

my_array = np.arange(1000)
reshape_10x10x10(my_array)
  '''(np.ndarray) -> np.ndarray
  Regresa un arreglo de 10x10x10.

  Parameters
  ----------
  arr: numpy.ndarray
   arreglo de numpy de 1000 elementos.

  Precondition
  ------------
    - arr.size == 1000
  '''


"""5. Create a NumPy array of 10000 numbers. Reshape the array into a 10x10x10x10 tensor."""

def reshape_10x10x10x10(arr):
    assert arr.size == 10000, 
  return arr.reshape(10, 10, 10, 10)

my_array = np.arange(10000)
reshape_10x10x10x10(my_array)
  '''(np.ndarray) -> np.ndarray
  Regresa un arreglo de numpy de 10x10x10x10.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 10000 elementos.

  Precondition
  ------------
    - arr.size == 10000
  '''
  

"""### NumPy Array Broadcasting

1. Add a NumPy array of shape (2, 3) to a NumPy array of shape (2, 1).
"""

def add_broadcast(arr1, arr2):
    return arr1 + arr2

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[10], [20]])
add_broadcast(arr1, arr2)
  '''Suma de dos arreglos de numpy con formas (2, 3) y (2, 1).

  Parameters
  ----------
  arr1: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  arr2: numpy.nd.array
    arreglo de numpy de forma (2, 1).
  '''
  

"""2. Subtract a NumPy array of shape (3, 2) from a NumPy array of shape (2, 3)."""

def subtract_broadcast(arr1, arr2):
    return arr1 - arr2.T

arr1 = np.array([[1, 2], [3, 4], [5, 6]])
arr2 = np.array([[10, 20, 30], [40, 50, 60]])
subtract_broadcast(arr1, arr2)
  '''(np.ndarray, np.ndarray) -> np.ndarray
  Regresa la resta de dos arreglos de numpy con formas (3, 2) y (2, 3).

  Parameters
  ----------
  arr1: numpy.ndarray
    arreglo de numpy de forma (3, 2).
  arr2: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  '''
  

"""3. Multiply a NumPy array of shape (2, 3) by a NumPy array of shape (3, 2)."""

def multiply_broadcast(arr1, arr2):
    return np.dot(arr1, arr2)

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8], [9, 10], [11, 12]])
multiply_broadcast(arr1, arr2
  '''Multiplica dos arreglos de numpy con formas (2, 3) y (3, 2).

  Parameters
  ---------
  arr1: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  arr2: numpy.ndarray
    arreglo de numpy de forma (3, 2).
  '''
  )

"""4. Divide a NumPy array of shape (2, 3) by a NumPy array of shape (2, 1)."""

def divide_broadcast(arr1, arr2):
    return arr1 / arr2

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[10], [20]])
divide_broadcast(arr1, arr2)
  '''Divide dos arreglos de numpy con formas (2, 3) y (2, 1).

  Parameters
  ----------
  arr1: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  arr2: numpy.ndarray
    arreglo de numpy de forma (2, 1).
  '''
  

"""5. Calculate the element-wise product of two NumPy arrays of shape (2, 3)."""

def element_wise_product(arr1, arr2):
    return arr1 * arr2

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
element_wise_product(arr1, arr2)
  '''Multiplica elemento a elemento dos arreglos de numpy con formas (2, 3).

  Parameters
  ----------
  arr1: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  arr2: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  '''
  

"""### Boolean Arrays and Masks

1. Temperature Data: You have a 1D NumPy array representing daily temperatures in Celsius.  Create a boolean mask that identifies days where the temperature was above 25 degrees Celsius.  Use this mask to print the temperatures on those days.  Also, calculate and print the number of days the temperature was below 15 degrees Celsius.
"""

def temp_data(temps):
    high_temps = temps[temps > 25]
  print(f"Temperaturas mayores a 25 grados: {high_temps}")
  low_temps_count = np.sum(temps < 15)
  print(f"Número de días con temperatura menor a 15 grados: {low_temps_count}")
    
temps = np.array([22, 28, 18, 30, 25, 15, 12, 20, 32, 26])
temp_data(temps)
    
  '''Imprime las temperaturas que fueron mayores a 25 grados y el número de
  días en los que la temperatura fue menor a 15 grados.

  Parameters
  ----------
  temps: numpy.ndarray
    arreglo de numpy de temperaturas en Celsius.
  '''
  

"""2. Rainfall Data: You have a 2D NumPy array representing monthly rainfall (in mm) for different cities.  Create a boolean mask to find the locations where rainfall exceeded 100 mm in any month.  Print the city indices (row numbers) that meet this condition."""

def rainfall_data(rainfall):
    indices = np.where(rainfall > 100)
    indices_lineales = np.ravel_multi_index(indices, rainfall.shape)
    print("Índices de las ciudades con más de 100 mm de lluvia:", indices_lineales)
    
    '''Imprime los índices de las ciudades que tuvieron más de 100 mm de lluvia.

    Parameters
    ----------
    rainfall: numpy.ndarray
        Arreglo 2D de numpy donde cada fila contiene datos de una ciudad.
    '''
    
    

"""3. Image Thresholding:  Imagine a grayscale image represented as a 2D NumPy array.  Create a mask to select pixels with intensity values greater than a certain threshold (e.g., 128).  Set the values of these pixels to 255 (white) and the remaining pixels to 0 (black). This simulates a simple image thresholding operation."""

def image_thresholding(image):
    binary_image = np.where(image >= threshold, 255, 0)
  return binary_image

threshold = 128
image = np.array([[100, 150, 200], [50, 120, 180], [20, 80, 140]])
image_thresholding(image)
  '''Genera un arreglo de numpy en blanco y negro.

  Parameters
  ----------
  image: numpy.ndarray
    arreglo 2D de numpy de una imagen en escala de grises.
  '''
  

"""### Fancy Indexing

1. Matrix Diagonals: Create a 5x5 matrix with values from 1 to 25.  Use fancy indexing to extract the elements on the main diagonal and the elements on the anti-diagonal.
"""

def matrix_diagonals(matrix):
    assert matrix.shape == (5, 5), 'La matriz debe ser de 5x5'
    main_diagonal = matrix[np.arange(5), np.arange(5)]
    invert_diagonal = matrix[np.arange(5), np.arange(4, -1, -1)]
  return main_diagonal, invert_diagonal

  '''Regresa un tuple con los elementos de la diagonal principal y antidiagonal.

  Parameters
  ----------
  matrix: numpy.ndarray
    arreglo 2D de numpy de 5x5.

  Precondition
  ------------
    - matrix.shape == (5, 5)
  '''
  
