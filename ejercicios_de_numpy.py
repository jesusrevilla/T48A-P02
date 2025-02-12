# -*- coding: utf-8 -*-
"""Ejercicios de numpy.ipynb
Automatically generated by Colab.
Original file is located at
    https://colab.research.google.com/drive/1u0A5L8EVWINhE-0G5AkgUdqdLWstQV1r
# Instrucciones
## Sigue las instrucciones para cada celda y al final baja la libreta en formato .py y subelo al repositorio que se te indique, no modifiques el nombre la de función, solo importa una vez numpy en el código.
"""

import numpy as np

"""### NumPy Arrays
@@ -36,11 +23,13 @@ def rand_int():
  '''
  return np.random.randint(0, 100, 10)

"""2. Create a NumPy array of 5 random floating-point numbers between 0 and 1."""

def rand_float():
  '''Regresa un arreglo de numpy con 5 números punto flotante entre 0 y 1.
  '''c
  Para poder mantener la generación de números aleatorios
  fija, en los ejemplos, se utiliza un seed.
@@ -55,7 +44,10 @@ def rand_float():
  >>> rand_float()
  array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701])
  '''
  return np.random.rand(5).astype(np.float64)

"""3. Create a NumPy array of the first 10 prime numbers."""

@@ -72,18 +64,27 @@ def first_10_primes():
  >>> first_10_primes()
  array([ 2,  3,  5,  7, 11, 13, 17, 19, 23, 29])
  '''
  return np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

first_10_primes()

"""4. Create a NumPy array of the squares of the numbers from 1 to 10."""

def squares():
  '''Regresa un arreglo de numpy con los cuadrados de los números del 1 al 10.
  '''
  return np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

squares()

"""5. Create a NumPy array of the cubes of the numbers from 1 to 10."""

def cubes():
  '''Regresa un arreglo de numpy con los cubos de los números del 1 al 10.
  '''
    return np.array([1, 8, 27, 64, 125, 216, 343, 512, 729, 1000])

cubes()

"""### NumPy Array Operations
@@ -118,18 +119,23 @@ def add_arrays(arr1, arr2):
  array([5, 7, 9])
  '''
  assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
  return arr1 + arr2

"""2. Subtract two NumPy arrays from each other, second argument less first."""

def subtract_arrays(arr1, arr2):
  '''Calcula arr2 menos arr1 (arreglos de numpy).
  '''
  assert arr1.shape == arr2.shape
  return arr2 - arr1

"""3. Multiply two NumPy arrays together (element-wise)."""

def multiply_arrays(arr1, arr2):
  '''Multiplica dos arreglos de numpy elemento por elemento.
  '''
  return arr1 * arr2

multiply_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))

"""4. Divide two NumPy arrays by each other (element-wise)."""

@@ -141,6 +147,9 @@ def divide_arrays(arr1, arr2):
    - arr2.any(0)
  '''
  assert arr1.any(0), 'No se puede dividir por cero'
  assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
  return arr2 / arr1

divide_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))

"""5. Create a NumPy array of the integer numbers from 1 to 5. Calculate the mean, median, and standard deviation."""

@@ -163,6 +172,11 @@ def stats(arr):
    - arr.size == 5
  '''
  assert arr.size == 5, 'El arreglo debe tener 5 elementos'
  return (np.mean(arr), np.median(arr), np.std(arr))


"""### NumPy Array Indexing and Slicing
@@ -183,6 +197,10 @@ def first_5(arr):
    - arr.size == 10
  '''
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[:5]

"""2. Create a NumPy array of 10 random integers between 0 and 100. Select the last 3 elements of the array."""

@@ -200,6 +218,10 @@ def last_3(arr):
    - arr.size == 10
  '''
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[-3:]


"""3. Create a NumPy array of 10 random integers between 0 and 100. Select the elements at indices 2, 4, and 6."""

@@ -217,6 +239,10 @@ def indices_2_4_6(arr):
    - arr.size == 10
  '''
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[[2, 4, 6]]


"""4. Create a NumPy array of 10 random integers between 0 and 100. Select the elements with values greater than 50."""

@@ -234,6 +260,10 @@ def greater_50(arr):
    - arr.size == 10
  '''
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[arr > 50]


"""5. Create a NumPy array of 10 random integers between 0 and 10. Select elements less than or equal to 7."""

@@ -250,7 +280,11 @@ def less_7(arr):
  ------------
    - arr.size == 10
  '''
  assert arr.size == 10, 'El arreglo (arr) debe tener 10 elementos'
  return arr[arr <= 7]

"""### NumPy Array Reshaping
@@ -270,6 +304,10 @@ def reshape_2x6(arr):
    - arr.size == 12
  '''
  assert arr.size == 12, 'El arreglo (arr) debe tener 12 elementos'
  return arr.reshape(2, 6)


"""2. Create a NumPy array of 24 numbers. Reshape the array into a 2x3x4 tensor."""

@@ -286,6 +324,10 @@ def reshape_2x3x4(arr):
    - arr.size == 24
  '''
  assert arr.size == 24, 'El arreglo (arr) debe tener 24 elementos'
  return arr.reshape(2, 3, 4)

"""3. Create a NumPy array of 100 numbers. Reshape the array into a 10x10 matrix."""

@@ -302,6 +344,10 @@ def reshape_10x10(arr):
    - arr.size == 100
  '''
  assert arr.size == 100, 'El arreglo (arr) debe tener 100 elementos'
  return arr.reshape(10, 10)

"""4. Create a NumPy array of 1000 numbers. Reshape the array into a 10x10x10 tensor."""

@@ -319,6 +365,10 @@ def reshape_10x10x10(arr):
    - arr.size == 1000
  '''
  assert arr.size == 1000, 'El arreglo (arr) debe tener 1000 elementos'
  return arr.reshape(10, 10, 10)

arr = np.random.randint(0, 100, 1000)
reshape_10x10x10(arr)

"""5. Create a NumPy array of 10000 numbers. Reshape the array into a 10x10x10x10 tensor."""

@@ -336,6 +386,10 @@ def reshape_10x10x10x10(arr):
    - arr.size == 10000
  '''
  assert arr.size == 10000, 'El arreglo (arr) debe tener 10000 elementos'
  return arr.reshape(10,10,10,10)

arr = np.random.randint(0, 100, 10000)
reshape_10x10x10x10(arr)

"""### NumPy Array Broadcasting
@@ -352,6 +406,12 @@ def add_broadcast(arr1, arr2):
  arr2: numpy.nd.array
    arreglo de numpy de forma (2, 1).
  '''
  return arr1 + arr2

arr1 = np.random.randint(1, 20, size=(2, 3))
arr2 = np.random.randint(1, 10, size=(2, 1))

add_broadcast(arr1, arr2)

"""2. Subtract a NumPy array of shape (3, 2) from a NumPy array of shape (2, 3)."""

@@ -366,6 +426,18 @@ def subtract_broadcast(arr1, arr2):
  arr2: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  '''
  #Para restar necesitamos que los dos arrays tengan las mismas dimensiones.

  if arr1.shape != arr2.shape: # Misma forma de array
    #Obtener transpuesta
    arr2 = arr2.T

  return arr1 - arr2

arr1 = np.random.randint(1, 20, size=(3, 2))
  arr2 = np.random.randint(1, 10, size=(2, 3))

  subtract_broadcast(arr1, arr2)

"""3. Multiply a NumPy array of shape (2, 3) by a NumPy array of shape (3, 2)."""

@@ -379,6 +451,12 @@ def multiply_broadcast(arr1, arr2):
  arr2: numpy.ndarray
    arreglo de numpy de forma (3, 2).
  '''
  return np.dot(arr1, arr2) # Realiza un producto matricial, esto implicando las relgas de multiplicacion de matrices

arr1 = np.random.randint(1, 20, size=(2, 3))
arr2 = np.random.randint(1, 10, size=(3, 2))

multiply_broadcast(arr1, arr2)

"""4. Divide a NumPy array of shape (2, 3) by a NumPy array of shape (2, 1)."""

@@ -392,6 +470,12 @@ def divide_broadcast(arr1, arr2):
  arr2: numpy.ndarray
    arreglo de numpy de forma (2, 1).
  '''
  return arr1 / arr2

arr1 = np.random.randint(1, 20, size=(2, 3))
arr2 = np.random.randint(1, 10, size=(2, 1))

divide_broadcast(arr1, arr2)

"""5. Calculate the element-wise product of two NumPy arrays of shape (2, 3)."""

@@ -405,6 +489,15 @@ def element_wise_product(arr1, arr2):
  arr2: numpy.ndarray
    arreglo de numpy de forma (2, 3).
  '''
  assert arr1.shape == arr2.shape # Misma forma
  assert arr1.shape == (2, 3) # Forma (2, 3), se asume que arr2 tiene la misma forma al pasar el assert pasado

  return arr1 * arr2

arr1 = np.random.randint(1, 20, size=(2, 3))
arr2 = np.random.randint(1, 10, size=(2, 3))

element_wise_product(arr1, arr2)

"""### Boolean Arrays and Masks
@@ -421,6 +514,17 @@ def temp_data(temps):
    arreglo de numpy de temperaturas en Celsius.
  '''

  mask_above_25 = temperatures >= 25 #Condicion mayor a 25 Grados celsius
  mask_below_15 = temperatures <= 15 #Condicion menor a 15 Grados celsius
  days = np.count_nonzero(mask_below_15)

  print(f"Temperaturas mayores a 25 grados: {temperatures[mask_above_25]}")
  print(f"Dias con temperaturas menores a 15 grados: {days}")

temperatures = np.array([22, 30, 18, 27, 12, 29, 15, 10, 25, 32])

temp_data(temperatures)

"""2. Rainfall Data: You have a 2D NumPy array representing monthly rainfall (in mm) for different cities.  Create a boolean mask to find the locations where rainfall exceeded 100 mm in any month.  Print the city indices (row numbers) that meet this condition."""

def rainfall_data(rainfall):
@@ -431,6 +535,13 @@ def rainfall_data(rainfall):
  rainfall: numpy.ndarray
    arreglo 2D de numpy de lluvia en mm y ciudades.
  '''
  mask_above_100 = rainfall >= 100
  cities = np.any(mask_above_100, axis=1)
  index_cities = np.where(cities)[0]
  return index_cities

rainfall = np.random.randint(30, 105, size=(4, 12))
rainfall_data(rainfall)

"""3. Image Thresholding:  Imagine a grayscale image represented as a 2D NumPy array.  Create a mask to select pixels with intensity values greater than a certain threshold (e.g., 128).  Set the values of these pixels to 255 (white) and the remaining pixels to 0 (black). This simulates a simple image thresholding operation."""

@@ -443,6 +554,14 @@ def image_thresholding(image):
    arreglo 2D de numpy de una imagen en escala de grises.
  '''

  mask = image > 128
  binary_image = np.zeros_like(image)
  binary_image[mask] = 255
  return binary_image

image = np.random.randint(0, 256, size=(5, 5))
image_thresholding(image)

"""### Fancy Indexing
1. Matrix Diagonals: Create a 5x5 matrix with values from 1 to 25.  Use fancy indexing to extract the elements on the main diagonal and the elements on the anti-diagonal.
@@ -461,5 +580,14 @@ def matrix_diagonals(matrix):
    - matrix.shape == (5, 5)
  '''
  assert matrix.shape == (5, 5), 'La matriz debe ser de 5x5'
  main_diagonal = matrix.diagonal()
  anti_diagonal = np.fliplr(matrix).diagonal()
  return main_diagonal, anti_diagonal

matrix = np.arange(1, 26).reshape(5, 5)
matrix_diagonals(matrix)

"""# Test"""

import doctest
doctest.testmod
