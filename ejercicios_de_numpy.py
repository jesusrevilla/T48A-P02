import numpy as np
import random

def rand_int():
  return np.random.randint(0,101,10)

np.random.seed(10)
rand_int()

def rand_float():
  np.random.seed(10)
  return np.random.rand(5)

np.random.seed(10)
rand_float().__repr__()

def first_10_primes():
  primos = []
  num = 2
  while len(primos) < 10:
      primo = all(num % p != 0 for p in primos)
      if primo:
          primos.append(num)
      num += 1
  return np.array(primos)

first_10_primes()

def squares():
  return np.array([i**2 for i in range(1, 11)])

squares()

def cubes():
  return np.array([i**3 for i in range(1, 11)])

cubes()

def add_arrays(arr1, arr2):
  assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
  return arr1 + arr2

add_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
add_arrays(arr1, arr2)

def subtract_arrays(arr1, arr2):
  assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
  return arr1 - arr2

subtract_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
subtract_arrays(arr1, arr2)

def multiply_arrays(arr1, arr2):
  assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
  return arr1 * arr2

multiply_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
multiply_arrays(arr1, arr2)

def divide_arrays(arr1, arr2):
  assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
  assert np.all(arr1 != 0), 'No se puede dividir por cero'
  return arr1 / arr2

divide_arrays(np.array([1, 2, 3]), np.array([4, 5, 6]))

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
divide_arrays(arr1, arr2)

def stats(arr):
  assert arr.size == 5, 'El arreglo debe tener 5 elementos'
  mean = np.mean(arr)
  median = np.median(arr)
  std_dev = np.std(arr)
  return (mean, median, std_dev)

arr = np.array([1, 2, 3, 4, 5])
stats(arr)

def first_5(arr):
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[:5]

random_array = np.random.randint(0, 100, 10)
print(random_array)
first_5(random_array)

def last_3(arr):
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[-3:]

random_array = np.random.randint(0, 100, 10)
print(random_array)
last_3(random_array)

def indices_2_4_6(arr):
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[[2, 4, 6]]

random_array = np.random.randint(0, 100, 10)
print(random_array)
indices_2_4_6(random_array)

def greater_50(arr):
  assert arr.size == 10, 'El arreglo debe tener 10 elementos'
  return arr[arr > 50]

random_array = np.random.randint(0, 100, 10)
print(random_array)
greater_50(random_array)

def less_7(arr):
  assert arr.size == 10, 'El arreglo (arr) debe tener 10 elementos'
  return arr[arr <= 7]

random_array = np.random.randint(0, 100, 10)
print(random_array)
less_7(random_array)

def reshape_2x6(arr):
  assert arr.size == 12, 'El arreglo (arr) debe tener 12 elementos'
  return arr.reshape(2, 6)

my_array = np.arange(12)
reshape_2x6(my_array)

def reshape_2x3x4(arr):
  assert arr.size == 24, 'El arreglo (arr) debe tener 24 elementos'
  return arr.reshape(2, 3, 4)

my_array = np.arange(24)
reshape_2x3x4(my_array)

def reshape_10x10(arr):
  assert arr.size == 100, 'El arreglo (arr) debe tener 100 elementos'
  return arr.reshape(10, 10)

sample_array = np.arange(100)
reshape_10x10(sample_array)

def reshape_10x10x10(arr):
  assert arr.size == 1000, 'El arreglo (arr) debe tener 1000 elementos'
  return arr.reshape(10, 10, 10)

my_array = np.arange(1000)
reshape_10x10x10(my_array)

def reshape_10x10x10x10(arr):
  assert arr.size == 10000, 'El arreglo (arr) debe tener 10000 elementos'
  return arr.reshape(10, 10, 10, 10)

my_array = np.arange(10000)
reshape_10x10x10x10(my_array)

def add_broadcast(arr1, arr2):
  return arr1 + arr2

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[10], [20]])
add_broadcast(arr1, arr2)

def subtract_broadcast(arr1, arr2):
  return arr1 - arr2.T

arr1 = np.array([[1, 2], [3, 4], [5, 6]])
arr2 = np.array([[10, 20, 30], [40, 50, 60]])
subtract_broadcast(arr1, arr2)

def multiply_broadcast(arr1, arr2):
  return np.dot(arr1, arr2)

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8], [9, 10], [11, 12]])
multiply_broadcast(arr1, arr2)

def divide_broadcast(arr1, arr2):
  return arr1 / arr2

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[10], [20]])
divide_broadcast(arr1, arr2)

def element_wise_product(arr1, arr2):
  return arr1 * arr2

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
element_wise_product(arr1, arr2)

def temp_data(temps):
  high_temps = temps[temps > 25]
  print(f"Temperaturas mayores a 25 grados: {high_temps}")
  low_temps_count = np.sum(temps < 15)
  print(f"Número de días con temperatura menor a 15 grados: {low_temps_count}")

temps = np.array([22, 28, 18, 30, 25, 15, 12, 20, 32, 26])
temp_data(temps)

def rainfall_data(rainfall):
    indices = np.where(rainfall > 100)
    indices_lineales = np.ravel_multi_index(indices, rainfall.shape)
    print("Índices de las ciudades con más de 100 mm de lluvia:", indices_lineales)

def image_thresholding(image):
  binary_image = np.where(image >= threshold, 255, 0)
  return binary_image

threshold = 128
image = np.array([[100, 150, 200], [50, 120, 180], [20, 80, 140]])
image_thresholding(image)

def matrix_diagonals(matrix):
  assert matrix.shape == (5, 5), 'La matriz debe ser de 5x5'
  main_diagonal = matrix[np.arange(5), np.arange(5)]
  anti_diagonal = matrix[np.arange(5), np.arange(4, -1, -1)]
  return main_diagonal, anti_diagonal
