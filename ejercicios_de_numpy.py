# -*- coding: utf-8 -*-
"""Ejercicios de NumPy: funciones solicitadas."""

import numpy as np


# ----------------------------
# NumPy Arrays
# ----------------------------

def rand_int():
    """
    Crea un arreglo de NumPy con 10 enteros aleatorios entre 0 y 100 (incluyente).

    Nota: no fija la semilla; se asume que quien llama puede usar np.random.seed.

    Returns
    -------
    numpy.ndarray
        Arreglo de 10 enteros entre 0 y 100.
    """
    return np.random.randint(0, 101, 10)


def rand_float():
    """
    Regresa un arreglo de NumPy con 5 números de punto flotante en [0, 1).

    Nota: no fija la semilla; se asume que quien llama puede usar np.random.seed.

    Returns
    -------
    numpy.ndarray
        Arreglo de 5 floats en [0, 1).
    """
    return np.random.rand(5)


def first_10_primes():
    """
    Crea un arreglo de NumPy con los diez primeros números primos.

    Returns
    -------
    numpy.ndarray
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    primos = []
    num = 2
    while len(primos) < 10:
        es_primo = True
        for p in primos:
            if num % p == 0:
                es_primo = False
                break
            if p * p > num:
                break
        if es_primo:
            primos.append(num)
        num += 1
    return np.array(primos, dtype=int)


def squares():
    """
    Regresa un arreglo de NumPy con los cuadrados de los números del 1 al 10.

    Returns
    -------
    numpy.ndarray
    """
    return np.array([i**2 for i in range(1, 11)], dtype=int)


def cubes():
    """
    Regresa un arreglo de NumPy con los cubos de los números del 1 al 10.

    Returns
    -------
    numpy.ndarray
    """
    return np.array([i**3 for i in range(1, 11)], dtype=int)


# ----------------------------
# NumPy Array Operations
# ----------------------------

def add_arrays(arr1, arr2):
    """
    Regresa la suma elemento a elemento de dos arreglos del mismo shape.
    """
    assert arr1.shape == arr2.shape, "arr1 y arr2 deben tener el mismo shape"
    return arr1 + arr2


def subtract_arrays(arr1, arr2):
    """
    Calcula arr1 - arr2 (arreglos de NumPy) elemento a elemento.
    """
    assert arr1.shape == arr2.shape, "arr1 y arr2 deben tener el mismo shape"
    return arr1 - arr2


def multiply_arrays(arr1, arr2):
    """
    Multiplica dos arreglos de NumPy elemento por elemento.
    """
    assert arr1.shape == arr2.shape, "arr1 y arr2 deben tener el mismo shape"
    return arr1 * arr2


def divide_arrays(arr1, arr2):
    """
    Divide elemento a elemento arr1 / arr2.

    Precondiciones:
      - arr1.shape == arr2.shape
      - arr2 no contiene ceros
    """
    assert arr1.shape == arr2.shape, "arr1 y arr2 deben tener el mismo shape"
    assert np.all(arr2 != 0), "arr2 no debe contener ceros para evitar división por cero"
    return arr1 / arr2


def stats(arr):
    """
    Calcula (media, mediana, desviacion_std) de un arreglo de tamaño 5.

    Returns
    -------
    tuple
        (media, mediana, desviacion_std)

    Precondición
    ------------
      - arr.size == 5
    """
    assert arr.size == 5, "El arreglo debe tener tamaño 5"
    mean = np.mean(arr)
    median = np.median(arr)
    std_dev = np.std(arr)
    return (mean, median, std_dev)


# ----------------------------
# NumPy Array Indexing and Slicing
# ----------------------------

def first_5(arr):
    """
    Regresa los primeros 5 elementos de un arreglo de tamaño 10.
    """
    assert arr.size == 10, "El arreglo debe tener 10 elementos"
    return arr[:5]


def last_3(arr):
    """
    Regresa los últimos 3 elementos de un arreglo de tamaño 10.
    """
    assert arr.size == 10, "El arreglo debe tener 10 elementos"
    return arr[-3:]


def indices_2_4_6(arr):
    """
    Regresa los elementos en los índices 2, 4 y 6 de un arreglo de tamaño 10.
    """
    assert arr.size == 10, "El arreglo debe tener 10 elementos"
    return arr[[2, 4, 6]]


def greater_50(arr):
    """
    Selecciona elementos mayores a 50 de un arreglo de tamaño 10.
    """
    assert arr.size == 10, "El arreglo debe tener 10 elementos"
    return arr[arr > 50]


def less_7(arr):
    """
    Selecciona elementos <= 7 de un arreglo de tamaño 10.
    """
    assert arr.size == 10, "El arreglo debe tener 10 elementos"
    return arr[arr <= 7]


# ----------------------------
# NumPy Array Reshaping
# ----------------------------

def reshape_2x6(arr):
    """
    Convierte un arreglo de 12 elementos a shape (2, 6).
    """
    assert arr.size == 12, "El arreglo debe tener 12 elementos"
    return arr.reshape(2, 6)


def reshape_2x3x4(arr):
    """
    Convierte un arreglo de 24 elementos a shape (2, 3, 4).
    """
    assert arr.size == 24, "El arreglo debe tener 24 elementos"
    return arr.reshape(2, 3, 4)


def reshape_10x10(arr):
    """
    Convierte un arreglo de 100 elementos a shape (10, 10).
    """
    assert arr.size == 100, "El arreglo debe tener 100 elementos"
    return arr.reshape(10, 10)


def reshape_10x10x10(arr):
    """
    Convierte un arreglo de 1000 elementos a shape (10, 10, 10).
    """
    assert arr.size == 1000, "El arreglo debe tener 1000 elementos"
    return arr.reshape(10, 10, 10)


def reshape_10x10x10x10(arr):
    """
    Convierte un arreglo de 10000 elementos a shape (10, 10, 10, 10).
    """
    assert arr.size == 10000, "El arreglo debe tener 10000 elementos"
    return arr.reshape(10, 10, 10, 10)


# ----------------------------
# NumPy Array Broadcasting
# ----------------------------

def add_broadcast(arr1, arr2):
    """
    Suma de dos arreglos con formas (2, 3) y (2, 1) usando broadcasting.
    """
    return arr1 + arr2


def subtract_broadcast(arr1, arr2):
    """
    Resta entre arreglos con formas (3, 2) y (2, 3) usando la traspuesta del segundo.
    Devuelve shape (3, 2).
    """
    return arr1 - arr2.T


def multiply_broadcast(arr1, arr2):
    """
    Multiplica matrices con formas (2, 3) y (3, 2) (producto matricial).
    """
    return arr1 @ arr2  # equivalente a np.dot(arr1, arr2)


def divide_broadcast(arr1, arr2):
    """
    Divide elemento a elemento arreglos con formas (2, 3) y (2, 1) usando broadcasting.
    """
    return arr1 / arr2


def element_wise_product(arr1, arr2):
    """
    Multiplica elemento a elemento dos arreglos de forma (2, 3).
    """
    return arr1 * arr2


# ----------------------------
# Boolean Arrays and Masks
# ----------------------------

def temp_data(temps):
    """
    Imprime temperaturas > 25 y el número de días con temperatura < 15.
    Además regresa (high_temps, low_temps_count) para facilitar pruebas.
    """
    high_temps = temps[temps > 25]
    print(f"Temperaturas mayores a 25 grados: {high_temps}")
    low_temps_count = int(np.sum(temps < 15))
    print(f"Número de días con temperatura menor a 15 grados: {low_temps_count}")
    return high_temps, low_temps_count


def rainfall_data(rainfall):
    """
    Imprime los índices (filas/ciudades) que tuvieron > 100 mm en algún mes.
    También regresa esos índices como un arreglo 1D único y ordenado.
    """
    rows = np.unique(np.where(rainfall > 100)[0])
    print("Índices de las ciudades con más de 100 mm de lluvia:", rows)
    return rows


def image_thresholding(image, threshold=128):
    """
    Umbraliza una imagen en escala de grises (2D).
    Valores >= threshold -> 255, el resto -> 0.
    """
    return np.where(image >= threshold, 255, 0)


# ----------------------------
# Fancy Indexing
# ----------------------------

def matrix_diagonals(matrix):
    """
    Regresa (diagonal_principal, antidiagonal) de una matriz 5x5.

    Precondición
    ------------
      - matrix.shape == (5, 5)
    """
    assert matrix.shape == (5, 5), "La matriz debe ser de 5x5"
    main_diagonal = matrix[np.arange(5), np.arange(5)]
    anti_diagonal = matrix[np.arange(5), np.arange(4, -1, -1)]
    return main_diagonal, anti_diagonal
