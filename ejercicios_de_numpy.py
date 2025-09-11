# -*- coding: utf-8 -*-
"""Ejercicios de numpy.ipynb
CUEVAS GONZALEZ ARELI ALEJANDRA 
 Sigue las instrucciones para cada celda y al final baja la libreta en formato .py y subelo al repositorio que se te indique, no modifiques el nombre la de función, solo importa una vez numpy en el código.
"""

import numpy as np
import random
# =============================
# NumPy Arrays
# =============================

def rand_int():
    """Crea un arreglo de numpy con 10 enteros aleatorios entre 0 y 100.

    Returns
    -------
    numpy.ndarray
        Arreglo de numpy con 10 enteros entre 0 y 100.
    """
    return np.random.randint(0, 101, 10)


def rand_float():
    """Regresa un arreglo de numpy con 5 números punto flotante entre 0 y 1.

    Returns
    -------
    numpy.ndarray
        Arreglo de numpy con 5 números punto flotante entre 0 y 1.
    """
    return np.random.rand(5)


def first_10_primes():
    """Crea un arreglo de numpy con los diez primeros números primos.

    Returns
    -------
    numpy.ndarray
        Arreglo de numpy con los diez primeros números primos.
    """
    primos = []
    num = 2
    while len(primos) < 10:
        if all(num % p != 0 for p in primos):
            primos.append(num)
        num += 1
    return np.array(primos)


def squares():
    """Regresa un arreglo con los cuadrados de los números del 1 al 10."""
    return np.array([i ** 2 for i in range(1, 11)])


def cubes():
    """Regresa un arreglo con los cubos de los números del 1 al 10."""
    return np.array([i ** 3 for i in range(1, 11)])


# =============================
# NumPy Array Operations
# =============================

def add_arrays(arr1, arr2):
    """Regresa la suma de dos arreglos de numpy (misma forma).

    Parameters
    ----------
    arr1, arr2 : numpy.ndarray
        Arreglos con la misma forma.
    """
    assert arr1.shape == arr2.shape, "arr1 y arr2 deben tener la misma forma"
    return arr1 + arr2


def subtract_arrays(arr1, arr2):
    """Resta elemento a elemento: arr1 - arr2.

    Parameters
    ----------
    arr1, arr2 : numpy.ndarray
        Arreglos con la misma forma.
    """
    assert arr1.shape == arr2.shape, "arr1 y arr2 deben tener la misma forma"
    return arr1 - arr2


def multiply_arrays(arr1, arr2):
    """Multiplica dos arreglos de numpy elemento por elemento."""
    assert arr1.shape == arr2.shape, "arr1 y arr2 deben tener la misma forma"
    return arr1 * arr2


def divide_arrays(arr1, arr2):
    """Divide elemento a elemento: arr1 / arr2.

    Precondiciones
    --------------
    - arr1.shape == arr2.shape
    - arr2 no contiene ceros
    """
    assert arr1.shape == arr2.shape, "arr1 y arr2 deben tener la misma forma"
    assert np.all(arr2 != 0), "arr2 no debe contener ceros"
    return arr1 / arr2


def stats(arr):
    """Calcula (media, mediana, desviación estándar) de un arreglo de 5 elementos.

    Parameters
    ----------
    arr : numpy.ndarray
        Arreglo de tamaño 5.

    Returns
    -------
    tuple
        (media, mediana, desviacion_std)
    """
    assert arr.size == 5, "arr debe tener tamaño 5"
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std_dev = float(np.std(arr))
    return (mean, median, std_dev)


# =============================
# Indexado y Slicing
# =============================

def first_5(arr):
    """Regresa los primeros 5 elementos de un arreglo de tamaño 10."""
    assert arr.size == 10, "arr debe tener tamaño 10"
    return arr[:5]


def last_3(arr):
    """Regresa los últimos 3 elementos de un arreglo de tamaño 10."""
    assert arr.size == 10, "arr debe tener tamaño 10"
    return arr[-3:]


def indices_2_4_6(arr):
    """Regresa los elementos en los índices 2, 4 y 6 de un arreglo de tamaño 10."""
    assert arr.size == 10, "arr debe tener tamaño 10"
    return arr[[2, 4, 6]]


def greater_50(arr):
    """Regresa los elementos > 50 de un arreglo de tamaño 10."""
    assert arr.size == 10, "arr debe tener tamaño 10"
    return arr[arr > 50]


def less_7(arr):
    """Regresa los elementos <= 7 de un arreglo de tamaño 10."""
    assert arr.size == 10, "arr debe tener tamaño 10"
    return arr[arr <= 7]


# =============================
# Reshaping
# =============================

def reshape_2x6(arr):
    """Convierte un arreglo de 12 elementos en forma (2, 6)."""
    assert arr.size == 12, "arr debe tener 12 elementos"
    return arr.reshape(2, 6)


def reshape_2x3x4(arr):
    """Convierte un arreglo de 24 elementos en forma (2, 3, 4)."""
    assert arr.size == 24, "arr debe tener 24 elementos"
    return arr.reshape(2, 3, 4)


def reshape_10x10(arr):
    """Convierte un arreglo de 100 elementos en forma (10, 10)."""
    assert arr.size == 100, "arr debe tener 100 elementos"
    return arr.reshape(10, 10)


def reshape_10x10x10(arr):
    """Convierte un arreglo de 1000 elementos en forma (10, 10, 10)."""
    assert arr.size == 1000, "arr debe tener 1000 elementos"
    return arr.reshape(10, 10, 10)


def reshape_10x10x10x10(arr):
    """Convierte un arreglo de 10000 elementos en forma (10, 10, 10, 10)."""
    assert arr.size == 10000, "arr debe tener 10000 elementos"
    return arr.reshape(10, 10, 10, 10)


# =============================
# Broadcasting / Álgebra
# =============================

def add_broadcast(arr1, arr2):
    """Suma con broadcasting (p.ej., formas (2,3) y (2,1))."""
    return arr1 + arr2


def subtract_broadcast(arr1, arr2):
    """Resta con broadcasting. Si arr1 es (3,2) y arr2 es (2,3), usa arr2.T."""
    return arr1 - arr2.T


def multiply_broadcast(arr1, arr2):
    """Multiplicación matricial: np.dot(arr1, arr2)."""
    return np.dot(arr1, arr2)


def divide_broadcast(arr1, arr2):
    """División con broadcasting (arr1 / arr2)."""
    assert np.all(arr2 != 0), "arr2 no debe contener ceros"
    return arr1 / arr2


def element_wise_product(arr1, arr2):
    """Producto elemento a elemento de dos arreglos con la misma forma."""
    assert arr1.shape == arr2.shape, "arr1 y arr2 deben tener la misma forma"
    return arr1 * arr2


# =============================
# Boolean Arrays y Masks
# =============================

def temp_data(temps):
    """Regresa temperaturas > 25 y el conteo de días con temperatura < 15.

    Parameters
    ----------
    temps : numpy.ndarray
        Temperaturas diarias en °C (1D).

    Returns
    -------
    tuple
        (temps_mayores_25, conteo_menores_15)
    """
    high_temps = temps[temps > 25]
    low_temps_count = int(np.sum(temps < 15))
    return high_temps, low_temps_count


def rainfall_data(rainfall):
    """Regresa los índices (filas) de ciudades con > 100 mm en algún mes.

    Parameters
    ----------
    rainfall : numpy.ndarray
        Matriz 2D donde cada fila corresponde a una ciudad y las columnas a meses.

    Returns
    -------
    numpy.ndarray
        Índices de filas donde alguna columna supera 100.
    """
    mask = (rainfall > 100).any(axis=1)
    return np.where(mask)[0]


def image_thresholding(image, threshold=128):
    """Binzariza una imagen en escala de grises usando un umbral.

    Valores >= threshold se asignan a 255; el resto a 0.

    Parameters
    ----------
    image : numpy.ndarray
        Imagen 2D en escala de grises.
    threshold : int, optional
        Umbral de binarización (default 128).

    Returns
    -------
    numpy.ndarray
        Imagen binaria (valores 0 o 255).
    """
    return np.where(image >= threshold, 255, 0)


# =============================
# Fancy Indexing
# =============================

def matrix_diagonals(matrix):
    """Regresa la diagonal principal y la antidiagonal de una matriz 5x5.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matriz 2D de forma (5, 5).

    Returns
    -------
    tuple
        (diagonal_principal, antidiagonal)
    """
    assert matrix.shape == (5, 5), "La matriz debe ser de 5x5"
    main_diagonal = matrix[np.arange(5), np.arange(5)]
    invert_diagonal = matrix[np.arange(5), np.arange(4, -1, -1)]
    return main_diagonal, invert_diagonal
