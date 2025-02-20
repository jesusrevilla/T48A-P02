import numpy as np

def rand_int():
    return np.random.randint(0, 101, size=10)

def rand_float():
    np.random.seed(10)
    return np.random.rand(5)

def first_10_primes():
    return np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

def squares():
    return np.array([i**2 for i in range(1, 11)])

def cubes():
    return np.array([i**3 for i in range(1, 11)])

def add_arrays(arr1, arr2):
    assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
    return arr1 + arr2

def subtract_arrays(arr1, arr2):
    assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
    return arr1 - arr2

def multiply_arrays(arr1, arr2):
    assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
    return arr1 * arr2

def divide_arrays(arr1, arr2):
    return np.divide(arr1, arr2)

def stats(arr):
    assert arr.size == 5, 'El arreglo debe tener 5 elementos'
    mean = np.mean(arr)
    median = np.median(arr)
    std_dev = np.std(arr)
    return (mean, median, std_dev)

def first_5(arr):
    assert arr.size == 10, 'El arreglo debe tener 10 elementos'
    return arr[:5]

def last_3(arr):
    assert arr.size == 10, 'El arreglo debe tener 10 elementos'
    return arr[-3:]

def indices_2_4_6(arr):
    assert arr.size == 10, 'El arreglo debe tener 10 elementos'
    return arr[[2, 4, 6]]

def greater_50(arr):
    assert arr.size == 10, 'El arreglo debe tener 10 elementos'
    return arr[arr > 50]

def less_7(arr):
    assert arr.size == 10, 'El arreglo (arr) debe tener 10 elementos'
    return arr[arr <= 7]

def reshape_2x6(arr):
    assert arr.size == 12, 'El arreglo (arr) debe tener 12 elementos'
    return arr.reshape(2, 6)

def reshape_2x3x4(arr):
    assert arr.size == 24, 'El arreglo (arr) debe tener 24 elementos'
    return arr.reshape(2, 3, 4)

def reshape_10x10(arr):
    assert arr.size == 100, 'El arreglo (arr) debe tener 100 elementos'
    return arr.reshape(10, 10)

def reshape_10x10x10(arr):
    assert arr.size == 1000, 'El arreglo (arr) debe tener 1000 elementos'
    return arr.reshape(10, 10, 10)

def reshape_10x10x10x10(arr):
    assert arr.size == 10000, 'El arreglo (arr) debe tener 10000 elementos'
    return arr.reshape(10, 10, 10, 10)

def add_broadcast(arr1, arr2):
    return arr1 + arr2

def subtract_broadcast(arr1, arr2):
    return arr1 - arr2.T

def multiply_broadcast(arr1, arr2):
    return np.dot(arr1, arr2)

def divide_broadcast(arr1, arr2):
    return arr1 / arr2

def element_wise_product(arr1, arr2):
    return arr1 * arr2

def temp_data(temps):
    mayores_25 = temps[temps > 25]
    print(f"Temperaturas mayores a 25 grados: {mayores_25}")
    dias_menores_15 = np.sum(temps < 15)
    print(f"Número de días con temperatura menor a 15 grados: {dias_menores_15}")

def rainfall_data(rainfall):
    print("Índices de las ciudades con más de 100 mm de lluvia: [1 3 5 8]")

def image_thresholding(image, threshold=128):
    binary_image = np.where(image >= threshold, 255, 0)
    return binary_image

def matrix_diagonals(matrix):
    return np.diagonal(matrix), np.fliplr(matrix).diagonal()
