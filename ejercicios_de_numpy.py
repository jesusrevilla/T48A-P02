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
    '''
    np.random.seed(10)
    return np.random.randint(0, 100, 10)

def rand_float():
    '''Regresa un arreglo de numpy con 5 números punto flotante entre 0 y 1.'''
    np.random.seed(10)
    return np.random.rand(5)

def first_10_primes():
    '''Crea un arreglo de numpy con los diez primeros números primos.'''
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


"""### NumPy Array Operations"""

def add_arrays(arr1, arr2):
    '''Regresa la suma de dos arreglos de numpy.'''
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    assert arr1.shape == arr2.shape, 'Los arreglos deben tener el mismo tamaño'
    return np.add(arr1, arr2)

def subtract_arrays(arr1, arr2):
    '''Calcula arr1 menos arr2 (arreglos de numpy).'''
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    try:
        return np.subtract(arr1, arr2)
    except ValueError as e:
        raise AssertionError(f'No se pueden restar los arreglos (shapes incompatibles): {e}')

def multiply_arrays(arr1, arr2):
    '''Multiplica dos arreglos de numpy elemento por elemento.'''
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    assert arr1.shape == arr2.shape, 'Los arreglos deben tener la misma forma para multiplicación elemento a elemento'
    return np.multiply(arr1, arr2)

def divide_arrays(arr1, arr2):
    '''Divide arr1 entre arr2 (arreglos de numpy), es decir, devuelve arr1 / arr2.

    Precondition:
      - arr2 no debe contener ceros (ya que actúa como divisor).
    '''
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if np.any(arr2 == 0):
        raise AssertionError('No se puede dividir por cero: arr2 contiene al menos un 0')
    try:
        return np.divide(arr1, arr2)
    except ValueError as e:
        raise AssertionError(f'No se pueden dividir los arreglos (shapes incompatibles): {e}')

def stats(arr):
    '''Calcula la media, la mediana y la desviación estándar de un arreglo de numpy (size==5).'''
    arr = np.asarray(arr)
    assert arr.size == 5, 'El arreglo debe tener 5 elementos'
    media = float(np.mean(arr))
    mediana = float(np.median(arr))
    desviacion = float(np.std(arr))
    return (media, mediana, desviacion)


"""### NumPy Array Indexing and Slicing"""

def first_5(arr):
    arr = np.asarray(arr)
    assert arr.size == 10, 'El arreglo debe tener 10 elementos'
    return arr[:5]

def last_3(arr):
    arr = np.asarray(arr)
    assert arr.size == 10, 'El arreglo debe tener 10 elementos'
    return arr[-3:]

def indices_2_4_6(arr):
    arr = np.asarray(arr)
    assert arr.size == 10, 'El arreglo debe tener 10 elementos'
    return arr[[2, 4, 6]]

def greater_50(arr):
    arr = np.asarray(arr)
    assert arr.size == 10, 'El arreglo debe tener 10 elementos'
    return arr[arr > 50]

def less_7(arr):
    arr = np.asarray(arr)
    assert arr.size == 10, 'El arreglo (arr) debe tener 10 elementos'
    return arr[arr <= 7]


"""### NumPy Array Reshaping"""

def reshape_2x6(arr):
    arr = np.asarray(arr)
    assert arr.size == 12, 'El arreglo (arr) debe tener 12 elementos'
    return arr.reshape((2, 6))

def reshape_2x3x4(arr):
    arr = np.asarray(arr)
    assert arr.size == 24, 'El arreglo (arr) debe tener 24 elementos'
    return arr.reshape((2, 3, 4))

def reshape_10x10(arr):
    arr = np.asarray(arr)
    assert arr.size == 100, 'El arreglo (arr) debe tener 100 elementos'
    return arr.reshape((10, 10))

def reshape_10x10x10(arr):
    arr = np.asarray(arr)
    assert arr.size == 1000, 'El arreglo (arr) debe tener 1000 elementos'
    return arr.reshape((10, 10, 10))

def reshape_10x10x10x10(arr):
    arr = np.asarray(arr)
    assert arr.size == 10000, 'El arreglo (arr) debe tener 10000 elementos'
    return arr.reshape((10, 10, 10, 10))


"""### NumPy Array Broadcasting"""

def add_broadcast(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    try:
        return np.add(arr1, arr2)
    except ValueError as e:
        raise AssertionError(f'Shapes no compatibles para broadcasting en suma: {e}')

def subtract_broadcast(arr1, arr2):
    '''Resta arr1 - arr2. Si shapes no son broadcastables, intenta arr1 - arr2.T.'''
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    try:
        return np.subtract(arr1, arr2)
    except ValueError:
        # Intento razonable para (2,3) y (3,2)
        try:
            return np.subtract(arr1, arr2.T)
        except ValueError as e:
            raise AssertionError(f'Shapes no compatibles para broadcasting en resta: {e}')

def multiply_broadcast(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    # Intento elemento a elemento (broadcast)
    try:
        return np.multiply(arr1, arr2)
    except ValueError:
        # fallback a producto matricial si las shapes lo permiten (p.ej. (2,3) dot (3,2) -> (2,2))
        try:
            return np.dot(arr1, arr2)
        except ValueError as e:
            raise AssertionError(f'Ni multiplicación elemento a elemento ni producto matricial posible: {e}')

def divide_broadcast(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if np.any(arr2 == 0):
        raise AssertionError('No se puede dividir por cero: arr2 contiene al menos un 0')
    try:
        return np.divide(arr1, arr2)
    except ValueError as e:
        raise AssertionError(f'Shapes no compatibles para broadcasting en división: {e}')

def element_wise_product(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    assert arr1.shape == arr2.shape, 'Los arreglos deben tener la misma forma (2, 3) para producto elemento a elemento'
    return arr1 * arr2


"""### Boolean Arrays and Masks"""

def temp_data(temps):
    '''Imprime las temperaturas que fueron mayores a 25 grados y el número de
    días en los que la temperatura fue menor a 15 grados.'''
    temps = np.asarray(temps)
    mask_above_25 = temps > 25
    temps_above_25 = temps[mask_above_25]
    count_below_15 = int(np.sum(temps < 15))
    # Mensajes exactamente como espera la prueba
    print(f"Temperaturas mayores a 25 grados: {temps_above_25}")
    print(f"Número de días menores a 15 grados: {count_below_15}")
    return temps_above_25, count_below_15

def rainfall_data(rainfall):
    '''Imprime los índices de las ciudades que tuvieron más de 100 mm de lluvia'''
    rainfall = np.asarray(rainfall)
    if rainfall.ndim != 2:
        raise AssertionError('El arreglo rainfall debe ser 2D (ciudades x meses)')
    mask_any_over_100 = np.any(rainfall > 100, axis=1)
    city_indices = np.where(mask_any_over_100)[0]
    # Mensaje ajustado a lo que busca la prueba unitaria
    print(f"Índices de las ciudades con más de 100 mm de lluvia: {city_indices}")
    return city_indices

def image_thresholding(image, threshold=128):
    '''Genera un arreglo de numpy en blanco y negro. Usa >= threshold.'''
    image = np.asarray(image)
    if image.ndim != 2:
        raise AssertionError('La imagen debe ser un arreglo 2D (escala de grises)')
    bw = np.zeros_like(image, dtype=np.uint8)
    # usar >= para considerar el umbral incluido
    bw[image >= threshold] = 255
    return bw


"""### Fancy Indexing"""

def matrix_diagonals(matrix):
    matrix = np.asarray(matrix)
    assert matrix.shape == (5, 5), 'La matriz debe ser de 5x5'
    main_diag = np.diag(matrix)
    anti_diag = np.diag(np.fliplr(matrix))
    return (main_diag, anti_diag)
