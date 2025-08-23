# Flores García Yahir Gerardo
# -*- coding: utf-8 -*-
import numpy as np

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
    np.random.seed(10)
    return np.random.randint(0, 100, 10)

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
    np.random.seed(10)
    return np.random.rand(5)

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
    lista_primos = []
    num = 2
    while len(lista_primos) < 10:
        is_primo = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_primo = False
                break
        if is_primo:
            lista_primos.append(num)
        num += 1
    return np.array(lista_primos)

def squares():
    '''Regresa un arreglo de numpy con los cuadrados de los números del 1 al 10.'''
    return np.array([i**2 for i in range(1, 11)])

def cubes():
    '''Regresa un arreglo de numpy con los cubos de los números del 1 al 10.'''
    return np.array([i**3 for i in range(1, 11)])
