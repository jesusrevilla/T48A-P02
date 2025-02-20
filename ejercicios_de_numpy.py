# -*- coding: utf-8 -*-
"""Copia de Ejercicios de numpy.ipynb
Gerardo Gael Elias Delgadillo - 177735
Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AkPZ-oIsACXggm8GT1sCXgR2qg9RIDMP

# Instrucciones

## Sigue las instrucciones para cada celda y al final baja la libreta en formato .py y subelo al repositorio que se te indique, no modifiques el nombre la de función, solo importa una vez numpy en el código.
"""

import numpy as np

"""### NumPy Arrays

1. Create a NumPy array of 10 random integers between 0 and 100.
"""

def rand_int():
  '''Crea un arreglo de numpy con 10 enteros aleatorios entre 0 y 100.
  Para poder mantener la generación de números aleatorios
  fija, en los ejemplos, se utiliza un seed.

  Returns
  -------
  numpy.ndarray
    Arreglo de numpy con 10 enteros aleatorios entre 0 y 100.

  Examples
  --------
  >>> np.random.seed(10)
  >>> rand_int()
  array([ 9, 15, 64, 28, 89, 93, 29,  8, 73,  0])
  '''
  return np.random.randint(0, 101, 10)

"""2. Create a NumPy array of 5 random floating-point numbers between 0 and 1."""

def rand_float():
  '''Regresa un arreglo de numpy con 5 números punto flotante entre 0 y 1.
  Para poder mantener la generación de números aleatorios
  fija, en los ejemplos, se utiliza un seed.

  Returns
  -------
  numpy.ndarray
    Arreglo de numpy con 5 números punto flotante entre 0 y 1.

  Examples
  --------
  >>> np.random.seed(10)
  >>> rand_float()
  array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701])
  '''
  return np.random.rand(5)

"""3. Create a NumPy array of the first 10 prime numbers."""

def first_10_primes():
  '''Crea un arreglo de numpy con los diez primeros números primos.

  Returns
  -------
  numpy.ndarray
    Arreglo de numpy con los diez primeros números primos.

  Examples
  --------
  >>> first_10_primes()
  array([ 2,  3,  5,  7, 11, 13, 17, 19, 23, 29])
  '''
  return np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

"""4. Create a NumPy array of the squares of the numbers from 1 to 10."""

def squares():
  '''Regresa un arreglo de numpy con los cuadrados de los números del 1 al 10.
  '''
  return np.array([x**2 for x in range(1, 11)])

"""5. Create a NumPy array of the cubes of the numbers from 1 to 10."""

def cubes():
  '''Regresa un arreglo de numpy con los cubos de los números del 1 al 10.
  '''
  return np.array([x**3 for x in range(1, 11)])

"""### NumPy Array Operations

1. Add two NumPy arrays together.
"""

def add_arrays(arr1, arr2):
  '''Regresa la suma de dos arreglos de numpy.

  Returns
  -------
  numpy.ndarray
    Suma de dos arreglos NumPy con el mismo tamaño.

  Parameters
  ----------
  arr1: numpy.ndarray
  arr2: numpy.ndarray

  Precondition
  ------------
    - arr1.shape == arr2.shape

  Examples
  --------
  >>> add_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))
  array([5, 7, 9])

  >>> arr1 = np.array([1, 2, 3])
  >>> arr2 = np.array([4, 5, 6])
  >>> add_arrays(arr1, arr2)
  array([5, 7, 9])
  '''
  assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
  return arr1 + arr2

"""2. Subtract two NumPy arrays from each other, second argument less first."""

def subtract_arrays(arr1, arr2):
  '''Calcula arr2 menos arr1 (arreglos de numpy).
  '''
  assert arr1.shape == arr2.shape,
  return arr2 - arr1

"""3. Multiply two NumPy arrays together (element-wise)."""

def multiply_arrays(arr1, arr2):
  '''Multiplica dos arreglos de numpy elemento por elemento.
  '''
  assert arr1.shape == arr2.shape,
  return arr1 * arr2

"""4. Divide two NumPy arrays by each other (element-wise)."""

def divide_arrays(arr1, arr2):
  '''Divide arr2 antre arr1 (arreglos de numpy).

  Precondition
  ------------
    - arr2.any(0)
  '''
  assert arr1.any(0), 'No se puede dividir por cero'
  return np.divide(arr1, arr2)

"""5. Create a NumPy array of the integer numbers from 1 to 5. Calculate the mean, median, and standard deviation."""

def stats(arr):
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
  assert arr.size == 5, 'El arreglo debe tener 5 elementos'
  return (np.mean(arr), np.median(arr), np.std(arr))

"""### NumPy Array Indexing and Slicing

1. Create a NumPy array of 10 random integers between 0 and 100. Select the first 5 elements of the array.
"""

def first_5(arr):
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
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[:5]

"""2. Create a NumPy array of 10 random integers between 0 and 100. Select the last 3 elements of the array."""

def last_3(arr):
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
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[-3:]

"""3. Create a NumPy array of 10 random integers between 0 and 100. Select the elements at indices 2, 4, and 6."""

def indices_2_4_6(arr):
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
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[[2, 4, 6]]

"""4. Create a NumPy array of 10 random integers between 0 and 100. Select the elements with values greater than 50."""

def greater_50(arr):
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
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[arr > 50]

"""5. Create a NumPy array of 10 random integers between 0 and 10. Select elements less than or equal to 7."""

def less_7(arr):
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
  assert arr.size == 10, 'El arreglo (arr) debe tener 10 elementos'
  return arr[arr <= 7]

"""### NumPy Array Reshaping

1. Create a NumPy array of 12 numbers. Reshape the array into a 2x6 matrix.
"""

def reshape_2x6(arr):
  '''Regresa un arreglo de numpy con 12 números y lo convierte en un arreglo de 2x6.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 12 elementos.

  Precondition
  ------------
    - arr.size == 12
  '''
  assert arr.size == 12, 'El arreglo (arr) debe tener 12 elementos'
  return arr.reshape(2, 6)

"""2. Create a NumPy array of 24 numbers. Reshape the array into a 2x3x4 tensor."""

def reshape_2x3x4(arr):
  '''Conviert un arreglo de numpy con 24 números en un arreglo de 2x3x4.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 24 elementos.

  Precondition
  ------------
    - arr.size == 24
  '''
  assert arr.size == 24, 'El arreglo (arr) debe tener 24 elementos'
  return arr.reshape(2, 3, 4)

"""3. Create a NumPy array of 100 numbers. Reshape the array into a 10x10 matrix."""

def reshape_10x10(arr):
  '''Convierte un numpy array en un numpy array de 10x10.

  Parameters
  ----------
  arr: numpy.ndarray
    arreglo de numpy de 100 elementos.

  Precondition
  ------------
    - arr.size == 100
  '''
  assert arr.size == 100, 'El arreglo (arr) debe tener 100 elementos'
  return arr.reshape(10, 10)

"""4. Create a NumPy array of 1000 numbers. Reshape the array into a 10x10x10 tensor."""

def reshape_10x10x10(arr):
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
  assert arr.size == 1000, 'El arreglo (arr) debe tener 1000 elementos'
  return arr.reshape(10, 10, 10)

"""5. Create a NumPy array of 10000 numbers. Reshape the array into a 10x10x10x10 tensor."""

def reshape_10x10x10x10(arr):
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
  assert arr.size == 10000, 'El arreglo (arr) debe tener 10000 elementos'
  return arr.reshape(10, 10, 10, 10)

"""### NumPy Array Broadcasting

1. Add a NumPy array of shape (2, 3) to a NumPy array of shape (2, 1).
"""

def add_broadcast(arr1, arr2):
  '''Suma de dos arreglos de numpy con formas (2, 3) y (2, 1).

  Parameters
  ----------
  arr1: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  arr2: numpy.nd.array
    arreglo de numpy de forma (2, 1).
  '''
  return arr1 + arr2

"""2. Subtract a NumPy array of shape (3, 2) from a NumPy array of shape (2, 3)."""

def subtract_broadcast(arr1, arr2):
  '''(np.ndarray, np.ndarray) -> np.ndarray
  Regresa la resta de dos arreglos de numpy con formas (3, 2) y (2, 3).

  Parameters
  ----------
  arr1: numpy.ndarray
    arreglo de numpy de forma (3, 2).
  arr2: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  '''
  return arr1 - arr2.T

"""3. Multiply a NumPy array of shape (2, 3) by a NumPy array of shape (3, 2)."""

def multiply_broadcast(arr1, arr2):
  '''Multiplica dos arreglos de numpy con formas (2, 3) y (3, 2).

  Parameters
  ---------
  arr1: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  arr2: numpy.ndarray
    arreglo de numpy de forma (3, 2).
  '''
  return np.dot(arr1, arr2)

"""4. Divide a NumPy array of shape (2, 3) by a NumPy array of shape (2, 1)."""

def divide_broadcast(arr1, arr2):
  '''Divide dos arreglos de numpy con formas (2, 3) y (2, 1).

  Parameters
  ----------
  arr1: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  arr2: numpy.ndarray
    arreglo de numpy de forma (2, 1).
  '''
  return arr1 / arr2

"""5. Calculate the element-wise product of two NumPy arrays of shape (2, 3)."""

def element_wise_product(arr1, arr2):
  '''Multiplica elemento a elemento dos arreglos de numpy con formas (2, 3).

  Parameters
  ----------
  arr1: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  arr2: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  '''
  return arr1 * arr2

"""### Boolean Arrays and Masks

1. Temperature Data: You have a 1D NumPy array representing daily temperatures in Celsius.  Create a boolean mask that identifies days where the temperature was above 25 degrees Celsius.  Use this mask to print the temperatures on those days.  Also, calculate and print the number of days the temperature was below 15 degrees Celsius.
"""

def temp_data(temps):
  '''Imprime las temperaturas que fueron mayores a 25 grados y el número de
  días en los que la temperatura fue menor a 15 grados.

  Parameters
  ----------
  temps: numpy.ndarray
    arreglo de numpy de temperaturas en Celsius.
  '''
  mayores_25 = temps[temps > 25]
  print(f"Temperaturas mayores a 25 grados: {mayores_25}")
  dias_menores_15 = np.sum(temps < 15)
  print(f"Días con temperatura menor a 15 grados: {dias_menores_15}")

"""2. Rainfall Data: You have a 2D NumPy array representing monthly rainfall (in mm) for different cities.  Create a boolean mask to find the locations where rainfall exceeded 100 mm in any month.  Print the city indices (row numbers) that meet this condition."""

def rainfall_data(rainfall):
  '''Imprime los índices de las ciudades que tuvieron más de 100 mm de lluvia

  Parameters
  ----------
  rainfall: numpy.ndarray
    arreglo 2D de numpy de lluvia en mm y ciudades.
  '''
  print("Índices de las ciudades con más de 100 mm de lluvia:", np.where(rainfall > 100)[0])

"""3. Image Thresholding:  Imagine a grayscale image represented as a 2D NumPy array.  Create a mask to select pixels with intensity values greater than a certain threshold (e.g., 128).  Set the values of these pixels to 255 (white) and the remaining pixels to 0 (black). This simulates a simple image thresholding operation."""

def image_thresholding(image, threshold=128):
  '''Genera un arreglo de numpy en blanco y negro.

  Parameters
  ----------
  image: numpy.ndarray
    arreglo 2D de numpy de una imagen en escala de grises.
  '''
  binary_image = np.where(image >= threshold, 255, 0)
  return binary_image

"""### Fancy Indexing

1. Matrix Diagonals: Create a 5x5 matrix with values from 1 to 25.  Use fancy indexing to extract the elements on the main diagonal and the elements on the anti-diagonal.
"""

def matrix_diagonals(matrix):
  '''Regresa un tuple con los elementos de la diagonal principal y antidiagonal.

  Parameters
  ----------
  matrix: numpy.ndarray
    arreglo 2D de numpy de 5x5.

  Precondition
  ------------
    - matrix.shape == (5, 5)
  '''
  assert matrix.shape == (5, 5)
  return np.diagonal(matrix), np.fliplr(matrix).diagonal()
