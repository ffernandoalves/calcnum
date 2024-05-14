"""
Created on Tue Feb 23 03:21:39 2021

@author: prof. Cleber Cavalcanti [#jesters]
"""

from typing import TypeVar, Union, Literal
from collections.abc import  Callable
from numbers import Number

import numpy as np
import matplotlib.pyplot as plt

# type Array = list[Number]
# type MathFunction = Callable[[Number], Number]

Array = TypeVar("Array", bound=list[Number])
MathFunction = TypeVar("MathFunction", bound=Callable[[Number], Number])


class mMathF[Number](Callable):
    pass

def particao(a: Number, b: Number, N: int) -> Array:
    """
    Primeira versão da função particao (escalar)

    Dados os pontos inicial $a$ e final $b$, e o número $N$
    de subintervalos da partição regular desejada, esta função
    retorna a partição regular com tais características, sob
    a forma de uma lista
    """
    lista = [a]
    delta = (b - a)/N
    for j in range(1, N+1):
        lista += [a + j * delta]
    return lista


def imagem(funcao: MathFunction, dominio: Array) -> Array:
    """
    Primeira versão da função imagem (escalar)

    Dadas uma função $funcao$ e uma lista $dominio$
    representando uma discretização do domínio da função
    $funcao$, esta função retorna uma discretização
    do conjunto imagem da função $funcao$, sob a forma de
    uma list
    """
    lista = []
    for x in dominio:
        lista += [funcao(x)]
    return lista


def grafico(funcao: MathFunction, dominio: Array) -> None:
    """
    Primeira versão da função grafico (uma função apenas)

    Dadas uma função $funcao$ e uma lista $dominio$
    representando uma discretização do domínio da função
    $funcao$, esta função exibe uma representação gráfica
    cartesiana do conjunto gráfico da função $funcao$ sob
    o dominio $dominio$
    """
    img = imagem(funcao, dominio)

    L = dominio[-1] - dominio[0]
    maxim, minim = max(img), min(img)
    H = maxim - minim
    # eixo das abscissas
    plt.plot([dominio[0] - 0.05*L, dominio[-1] + 0.05*L],
             [minim - 0.025*H, minim - 0.025*H],
             'k-', lw=0.5)

    # eixo das ordenadas
    plt.plot([dominio[0] - 0.025*L, dominio[0] - 0.025*L],
             [minim - 0.05*H, maxim + 0.05*H],
             'k-', lw=0.5)

    # grafico propriamente dito
    plt.plot(dominio, img, 'b-', lw=2)


def erro_abs(v_calculado, v_verdadeiro):
    '''
    Erro Absoluto
    '''
    return abs(v_calculado - v_verdadeiro)


def erro_rel(v_calculado, v_verdadeiro):
    '''
    Erro Relativo
    '''
    if v_verdadeiro != 0:
        saida = abs(1 - v_calculado/v_verdadeiro)
    elif v_verdadeiro == 0:
        print('Erro! v_verdeiro é zero.')
        saida = False
    return saida


def any2vector(objeto: Union[Array, np.ndarray]) -> Union[np.ndarray, Array]:
    '''
    Transforma lista em vetor, matriz linha em vetor, e matriz coluna em vetor
    '''
    if isinstance(objeto, list):
        objeto_aux = np.array(objeto, dtype=float)
    elif isinstance(objeto, np.ndarray):
        # Matrix 1 X 1
        if np.shape(objeto) == (1, 1):
            objeto_aux = objeto[0]
        # Matriz linha
        elif np.shape(objeto)[0] == 1:
            objeto_aux = objeto[0]
        # Vetor
        elif len(np.shape(objeto)) == 1:
            objeto_aux = objeto
        # Matriz coluna
        elif np.shape(objeto)[1] == 1:
            objeto_aux = (objeto.T)[0]
    # Objeto qualquer
    elif isinstance(objeto, type(particao)):
        objeto_aux = objeto
    else:
        print('any2vector: Objeto -> Objeto')
        objeto_aux = objeto
    return objeto_aux


def func2vector(funcao: Callable, dominio: Array) -> Union[Array, np.ndarray, Literal[False]]:
    '''
    Transforma função ou lista que representa imagem de função
    em um vetor que representa a imagem da função dada
    '''
    temp1 = any2vector(funcao)
    temp2 = any2vector(dominio)

    if isinstance(funcao, type(particao)):
        imf = imagem(funcao, dominio)
    elif np.shape(temp1) != np.shape(temp2):
        imf = False
    else:
        imf = temp1

    return imf

######################
# RAÍZES DE EQUAÇÕES #
######################


def bisseccao(funcao, xmin, xmax):
    '''
    Método da Bissecção para determinar raiz da equação
    f(x) = 0 para x entre xmin e xmax

    Retorna uma lista com as aproximações
    '''

    tol = 10**(-8)

    fxmin = funcao(xmin)
    fxmax = funcao(xmax)

    xmeio = 0.5 * (xmin + xmax)
    aprox = [xmeio]
    fxmei = funcao(xmeio)

    if fxmin == 0:
        aprox += [xmin]
        return aprox

    elif fxmax == 0:
        aprox += [xmax]
        return aprox

    elif (fxmin > 0 and fxmax > 0) or (fxmin < 0 and fxmax < 0):
        print('Erro! f(xmin) e f(xmax) possuem o mesmo sinal')
        return False

    while abs(xmin - xmax) >= tol:  # abs(fxmei) >= tol:

        if (fxmei > 0 and fxmin > 0) or (fxmei < 0 and fxmin < 0):
            xmin = xmeio

        elif (fxmei > 0 and fxmax > 0) or (fxmei < 0 and fxmax < 0):
            xmax = xmeio

        elif fxmei == 0:
            return aprox

        xmeio = 0.5 * (xmin + xmax)
        aprox += [xmeio]
        fxmei = funcao(xmeio)

    return aprox


def pontofixo(funcao, dominio, k, MAX):
    '''
    Método do Ponto Fixo

    Intenciona-se resolver a equação
        funcao(x) = 0, com x in dominio

    ou equivalentemente,
        g(x)= x - k * funcao(x) = x, com x in dominio

    Retorna uma lista com as aproximações
    '''

    tol = 10**(-8)

    medio = 0.5 * (dominio[0] + dominio[-1])
    if abs(funcao(medio)) < tol:
        return [medio]

    def g(x):
        return x - k*funcao(x)

    if g(medio) < dominio[0] or g(medio) > dominio[-1]:
        print('Erro! Não há ponto fixo.')
        aprox = False

    elif g(medio) >= dominio[0] and g(medio) <= dominio[-1]:

        aprox = [medio, g(medio)]

        num = 3  # numero de iteracoes
        while abs(funcao(aprox[-1])) >= tol and num <= MAX:
            aprox += [g(aprox[-1])]
            num += 1

    else:
        print('Erro crasso!')
        aprox = False

    return aprox


def newtonraphson(funcao, inicial, tol):
    '''
    Método Newton-Raphson

    Resolve a equação funcao(x) = 0

    Retorna uma lista com as aproximações

    Cuidado! O presente código funciona se a função não oscilar muito
    '''

    aprox = [inicial]
    if funcao(aprox[0]) == 0:
        return aprox

    h = 0.01
    dfmei = 0.5 * (funcao(aprox[0] + h) - funcao(aprox[0] - h))/h
    aprox += [aprox[0] - funcao(aprox[0])/dfmei]

    num = 3
    while abs(aprox[-2] - aprox[-1]) >= tol:
        if num == 15:
            print('Erro! A aproximação diverge')
            return False

        if funcao(aprox[-1]) == 0:
            return aprox

        h = 0.01 * abs(aprox[-2] - aprox[-1])
        dfmei = 0.5 * (funcao(aprox[-1] + h) - funcao(aprox[-1] - h))/h
        aprox += [aprox[-1] - funcao(aprox[-1])/dfmei]

        num += 1

    return aprox


def regulafalsi(funcao, dominio, tol):
    '''
    Regula Falsi - Retorna uma lista com as aproximações
    '''
    x1 = dominio[0]
    x2 = dominio[-1]
    fx1 = funcao(x1)
    fx2 = funcao(x2)

    aprox = [x1, x2]

    if fx1 == 0:
        aprox += [x1]
        return aprox

    elif fx2 == 0:
        aprox += [x2]
        return aprox

    elif (fx1 > 0 and fx2 > 0) or (fx1 < 0 and fx2 < 0):
        return 'Erro! A função não muda de sinal no intervalo especificado'

    plt.plot([x1, x2], [fx1, fx2], 'r:', lw=1)
    plt.plot([x1, x2], [fx1, fx2], 'o', color='orange', lw=0.5,
             markersize=2)

    niter = 1
    while abs(aprox[-2] - aprox[-1]) >= tol and fx1 != fx2:
        x3 = x2 - fx2 * (x1 - x2)/(fx1 - fx2)
        fx3 = funcao(x3)

        print('x1 =', x1, '\b,  x2 =', x2)
        print('|x1 - x2| =', abs(x1 - x2))
        print('x['+str(niter)+'] =', x3,
              '\b,  f(x['+str(niter)+']) =', fx3, '\n')
        plt.plot([x3, x3], [0, fx3], 'r:', lw=1)

        niter += 1
        aprox += [x3]

        if fx3 == 0:
            return aprox

        elif (fx3 > 0 and fx1 > 0) or (fx3 < 0 and fx1 < 0):
            x1, fx1 = x3, fx3

        elif (fx3 > 0 and fx2 > 0) or (fx3 < 0 and fx2 < 0):
            x2, fx2 = x3, fx3

        plt.plot([x1, x2], [fx1, fx2], 'r:', lw=1)
        plt.plot([x1, x2], [fx1, fx2], 'o', color='orange', lw=0.5,
                 markersize=2)

    plt.plot([aprox[-1]], [funcao(aprox[-1])], 'ro', markersize=2)

    return aprox


def secantes(funcao, dominio, tol):
    '''
    Método das Secantes - Retorna uma lista com as aproximações
    '''
    x1 = dominio[0]
    x2 = dominio[-1]
    fx1 = funcao(x1)
    fx2 = funcao(x2)

    aprox = [x1, x2]

    if fx1 == 0:
        aprox += [x1]
        return aprox

    elif fx2 == 0:
        aprox += [x2]
        return aprox

    numiter = 3
    while abs(aprox[-2] - aprox[-1]) >= tol and fx1 != fx2:
        x3 = x2 - fx2 * (x1 - x2)/(fx1 - fx2)

        if funcao(x3) == 0:
            aprox += [x3]
            return aprox

        aprox += [x3]
        """
        if (x3 > x1 and x1 > x2) or (x3 < x1 and x1 < x2):
            plt.plot([x2, x3], [fx2, 0], 'r:', lw=0.5)
        elif (x3 > x2 and x2 > x1) or (x3 < x2 and x2 < x1):
            plt.plot([x1, x3], [fx1, 0], 'r:', lw=0.5)
        elif (x3 < abs(x1 - x2)):
            plt.plot([x1, x2], [fx1, fx2], 'r:', lw=0.5)
        plt.plot([x3, x3], [0, funcao(x3)], 'r:', lw=0.5)
        plt.plot([x1, x2], [fx1, fx2], 'o', color='orange', markersize=2)
        print('iteracao =', numiter)
        """
        x1, fx1 = x2, fx2
        x2, fx2 = x3, funcao(x3)

        # print('f(x1) = ', fx1, '\b,  f(x2) =', fx2, '\n')
        numiter += 1

    return aprox


#############
# DERIVAÇÃO #
#############


def dfmais(funcao, x, h):
    '''
    Derivada progressiva (Derivada à direita) - erro da ordem de h
    '''
    return (funcao(x + h) - funcao(x)) / h


def dfmenos(funcao, x, h):
    '''
    Derivada retrograda (Derivada à esquerda) - erro da ordem de h
    '''
    return (funcao(x) - funcao(x - h)) / h


def dfcentro(funcao, x, h):
    '''
    Derivada central - erro da ordem de h**2
    '''
    return 0.5 * (funcao(x + h) - funcao(x - h)) / h


def d2f(funcao, x, h):
    '''
    Derivada segunda (central) - erro da ordem de h**2
    '''
    return (funcao(x - h) - 2 * funcao(x) + funcao(x + h)) / (h**2)


def df5p(funcao, x, h):
    '''
    Derivada, 5 pontos (central) - erro da ordem de h**4
    '''
    return (funcao(x - 2*h) - 8 * funcao(x - h) + 8 * funcao(x + h)
            - funcao(x + 2*h)) / (12 * h)


def d2f5p(funcao, x, h):
    '''
    Derivada segunda (central), 5 pontos - erro da ordem de h**4
    '''
    return (- funcao(x - 2*h) + 16 * funcao(x - h) - 30 * funcao(x)
            + 16 * funcao(x + h) - funcao(x + 2*h)) / (12 * h**2)


def d3f5p(funcao, x, h):
    '''
    Derivada terceira (central), 5 pontos - erro da ordem de h**2
    '''
    return 0.5 * (- funcao(x - 2*h) + 2*funcao(x - h) - 2*funcao(x + h)
                  + funcao(x + 2*h)) / h**3


def d4f5p(funcao, x, h):
    '''
    Derivada quarta (central), 5 pontos - erro da ordem de h**2
    '''
    return (funcao(x - 2*h) - 4 * funcao(x - h) + 6 * funcao(x)
            - 4 * funcao(x + h) + funcao(x + 2*h)) / (h**4)


def dfmais3p(funcao, x, h):
    '''
    Derivada à direita, 3 pontos - erro da ordem de h**2
    '''
    return 0.5 * (- 3 * funcao(x) + 4 * funcao(x + h) - funcao(x + 2*h)) / h


def d2fmais3p(funcao, x, h):
    '''
    Derivada segunda à direita, 3 pontos - erro da ordem de h
    '''
    return (funcao(x) - 2 * funcao(x + h) + funcao(x + 2*h)) / (h**2)


def dfmais4p(funcao, x, h):
    '''
    Derivada à direita, 4 pontos - erro da ordem de h**3
    '''
    return (- 11 * funcao(x) + 18 * funcao(x + h) - 9 * funcao(x + 2*h)
            + 2 * funcao(x + 3*h)) / (6*h)


def d2fmais4p(funcao, x, h):
    '''
    Derivada segunda à direita, 4 pontos -  erro da ordem de h**2
    '''
    return (2*funcao(x) - 5*funcao(x+h) + 4*funcao(x+2*h) - funcao(x+3*h))/h**2


def d3fmais4p(funcao, x, h):
    '''
    Derivada terceira à esquerda, 4 pontos - erro da ordem e h
    '''
    return (- funcao(x) + 18*funcao(x + h) - 18*funcao(x + 2*h)
            + funcao(x + 3*h)) / h**3


def dfmais5p(funcao, x, h):
    '''
    Derivada à direita, 5 pontos - erro da ordem de h**4
    '''
    return (- 25*funcao(x) + 48*funcao(x + h) - 36*funcao(x + 2*h)
            + 16*funcao(x + 3*h) - 3*funcao(x + 4*h))/(12 * h)


def dfmenos3p(funcao, x, h):
    '''
    Derivada à esquerda, 3 pontos - erro da ordem de h**2
    '''
    return 0.5 * (funcao(x - 2*h) - 4 * funcao(x - h) + 3 * funcao(x)) / h


def d2fmenos3p(funcao, x, h):
    '''
    Derivada segunda à esquerda, 3 pontos - erro da ordem de h
    '''
    return (funcao(x - 2*h) - 2 * funcao(x - h) + funcao(x)) / (h**2)


def dfmenos4p(funcao, x, h):
    '''
    Derivada à esquerda, 4 pontos - erro da ordem de h**3
    '''
    return (11 * funcao(x) - 18 * funcao(x - h) + 9 * funcao(x - 2*h)
            - 2 * funcao(x - 3*h)) / (6 * h)


def d2fmenos4p(funcao, x, h):
    '''
    Derivada segunda à esquerda, 4 pontos -  erro da ordem de h**2
    '''
    return (2 * funcao(x) - 5 * funcao(x - h) + 4 * funcao(x - 2*h)
            - funcao(x - 3*h)) / h**2


def d3fmenos4p(funcao, x, h):
    '''
    Derivada terceira à esquerda, 4 pontos - erro da ordem de h
    '''
    return (funcao(x) - 18*funcao(x - h) + 18*funcao(x - 2*h)
            - funcao(x - 3*h)) / h**3


def dfmenos5p(funcao, x, h):
    '''
    Derivada à esquerda, 5 pontos - erro da ordem de h**4
    '''
    return 0.25 * (25 * funcao(x) - 48 * funcao(x - h) + 36 * funcao(x - 2*h)
                   - 16 * funcao(x - 3*h) + 3 * funcao(x - 4*h)) / (3*h)


def funcao_df3p(funcao, dominio):
    '''
    Dada uma função (ou a imagem de uma função) sobre um intervalo
    (partição regular), retorna a imagem da função derivada

    Erro da ordem de h**2
    '''

    dominio = any2vector(dominio)
    h = dominio[1] - dominio[0]
    df3p = np.empty((len(dominio), ), dtype=float)

    imf = func2vector(funcao, dominio)

    if np.shape(dominio)[0] < 3:
        print('Não há pontos suficientes no domínio')
        df3p = False

    else:
        df3p[0] = 0.5 * (- 3 * imf[0] + 4 * imf[1] - imf[2]) / h
        for i in range(1, len(dominio) - 1):
            df3p[i] = 0.5 * (imf[i + 1] - imf[i - 1]) / h
        df3p[len(dominio) - 1] = 0.5*(imf[len(dominio) - 3]
                                      - 4*imf[len(dominio) - 2]
                                      + 3*imf[len(dominio) - 1]) / h

    return df3p


def funcao_df5p(funcao, dominio):
    '''
    Dada uma função (ou a imagem de uma função) sobre um intervalo
    (partição regular), retorna a imagem da função derivada

    Erro da ordem de h**4
    '''
    dominio = any2vector(dominio)
    h = dominio[1] - dominio[0]
    df5p = np.empty((len(dominio), ), dtype=float)

    imf = func2vector(funcao, dominio)

    if np.shape(dominio)[0] < 5:
        print('Não há pontos suficientes no domínio')
        df5p = False

    else:
        df5p[0] = 0.25 * (- 25 * imf[0] + 48 * imf[1] - 36 * imf[2]
                          + 16 * imf[3] - 3 * imf[4]) / (3*h)
        df5p[1] = 0.25 * (- 3 * imf[0] - 10 * imf[1] + 18 * imf[2]
                          - 6 * imf[3] + imf[4]) / (3*h)
        for i in range(2, len(dominio) - 2):
            df5p[i] = (imf[i - 2] - 8 * imf[i - 1] + 8 * imf[i + 1]
                       - imf[i + 2]) / (12 * h)
        df5p[-2] = 0.25 * (3 * imf[-1] + 10 * imf[-2] - 18 * imf[-3]
                           + 6 * imf[-4] - imf[-5]) / (3*h)
        df5p[-1] = 0.25 * (25 * imf[-1] - 48 * imf[-2] + 36 * imf[-3]
                           - 16 * imf[-4] + 3 * imf[-5]) / (3*h)

    return df5p

##############
# INTEGRAÇÃO #
##############


def intmidpoint(funcao, dominio):
    '''
    Método do Ponto Médio - particao qualquer
    '''
    integral = 0
    for j in range(0, len(dominio) - 1):
        medio = 0.5 * (dominio[j] + dominio[j+1])
        integral += funcao(medio) * (dominio[j+1] - dominio[j])
    return integral


def trapezio(funcao, dominio):
    '''
    Método do Trapézio - particao qualquer
    '''
    integral = 0
    for j in range(0, len(dominio) - 1):
        inicio_j, final_j = dominio[j], dominio[j+1]
        delta = final_j - inicio_j
        integral += 0.5*delta*(funcao(final_j) + funcao(inicio_j))
    return integral


def simpson(funcao, dominio):
    '''
    Regra de Simpson - particao qualquer
    '''
    integral = 0
    for j in range(0, len(dominio) - 1):
        inicio_j, final_j = dominio[j], dominio[j+1]
        delta = final_j - inicio_j
        fmeio = funcao(0.5*(inicio_j + final_j))
        integral += 0.5*delta*(funcao(inicio_j) + 4*fmeio + funcao(final_j))/3
    return integral


def simpson38(funcao, dominio):
    '''
    Regra de Simpson 3/8 - particao qualquer
    '''
    integral = 0
    for j in range(0, len(dominio) - 1):
        inicio_j, final_j = dominio[j], dominio[j+1]
        delta = final_j - inicio_j
        fterco1 = funcao((2*inicio_j + final_j)/3)
        fterco2 = funcao((inicio_j + 2*final_j)/3)
        integral += 0.125 * delta * (funcao(inicio_j) + 3*(fterco1 + fterco2)
                                     + funcao(final_j))
    return integral


def boole(funcao, dominio):
    '''
    Regra de Boole - particao qualquer
    '''
    integral = 0
    for j in range(0, len(dominio) - 1):
        inicio_j, final_j = dominio[j], dominio[j+1]
        delta = final_j - inicio_j
        fquarto1 = funcao(0.25*(3*inicio_j + final_j))
        fquarto2 = funcao(0.5*(inicio_j + final_j))
        fquarto3 = funcao(0.25*(inicio_j + 3*final_j))
        integral += 0.1 * delta * (7*(funcao(inicio_j) + funcao(final_j))
                                   + 32*(fquarto1 + fquarto3)
                                   + 12*fquarto2)/9
    return integral


def int6pclosed(funcao, dominio):
    '''
    Newton-Cotes fechada - 6 pontos
    particao qualquer
    '''
    integral = 0
    for j in range(0, len(dominio) - 1):
        inicio_j, final_j = dominio[j], dominio[j+1]
        delta = final_j - inicio_j
        f_5_1 = funcao(0.2*(4*inicio_j + final_j))
        f_5_2 = funcao(0.2*(3*inicio_j + 2*final_j))
        f_5_3 = funcao(0.2*(2*inicio_j + 3*final_j))
        f_5_4 = funcao(0.2*(inicio_j + 4*final_j))
        integral += 0.03125 * delta * (19*(funcao(inicio_j) + funcao(final_j))
                                       + 75*(f_5_4 + f_5_1)
                                       + 50*(f_5_2 + f_5_3))/9
    return integral


def int7pclosed(funcao, dominio):
    '''
    Newton-Cotes fechada - 7 pontos
    particao qualquer
    '''
    integral = 0
    for j in range(0, len(dominio) - 1):
        inicio_j, final_j = dominio[j], dominio[j+1]
        delta = final_j - inicio_j
        f_6_1 = funcao(0.5*(5*inicio_j + final_j)/3)
        f_6_2 = funcao((2*inicio_j + final_j)/3)
        f_6_3 = funcao(0.5*(inicio_j + final_j))
        f_6_4 = funcao((inicio_j + 2*final_j)/3)
        f_6_5 = funcao(0.5*(inicio_j + 5*final_j)/3)
        integral += 0.025 * delta * (41*(funcao(inicio_j) + funcao(final_j))
                                     + 27*(f_6_2 + f_6_4)
                                     + 216*(f_6_5 + f_6_1)
                                     + 272*f_6_3)/21
    return integral


def int8pclosed(funcao, dominio):
    '''
    Newton-Cotes fechada - 8 pontos
    particao qualquer
    '''
    integral = 0
    for j in range(0, len(dominio) - 1):
        inicio_j, final_j = dominio[j], dominio[j+1]
        delta = final_j - inicio_j
        f_7_1 = funcao((6*inicio_j + final_j)/7)
        f_7_2 = funcao((5*inicio_j + 2*final_j)/7)
        f_7_3 = funcao((4*inicio_j + 3*final_j)/7)
        f_7_4 = funcao((3*inicio_j + 4*final_j)/7)
        f_7_5 = funcao((2*inicio_j + 5*final_j)/7)
        f_7_6 = funcao((inicio_j + 6*final_j)/7)
        integral += 0.0015625 * delta*(751*(funcao(inicio_j) + funcao(final_j))
                                       + 1323*(f_7_2 + f_7_5)
                                       + 2989*(f_7_4 + f_7_3)
                                       + 3577*(f_7_1 + f_7_6))/27
        # 1/17280 = 1/(64 * 27 * 10) = 0.0015625 * (1/27)
    return integral


def int9pclosed(funcao, dominio):
    '''
    Newton-Cotes fechada - 8 pontos
    particao qualquer
    '''
    integral = 0
    for j in range(0, len(dominio) - 1):
        inicio_j, final_j = dominio[j], dominio[j+1]
        delta = final_j - inicio_j
        f_8_1 = funcao(0.125*(7*inicio_j + final_j))
        f_8_2 = funcao(0.125*(6*inicio_j + 2*final_j))
        f_8_3 = funcao(0.125*(5*inicio_j + 3*final_j))
        f_8_4 = funcao(0.125*(4*inicio_j + 4*final_j))
        f_8_5 = funcao(0.125*(3*inicio_j + 5*final_j))
        f_8_6 = funcao(0.125*(2*inicio_j + 6*final_j))
        f_8_7 = funcao(0.125*(inicio_j + 7*final_j))
        integral += 0.02 * delta * (989*(funcao(inicio_j) + funcao(final_j))
                                    + 5888*(f_8_1 + f_8_7)
                                    - 928*(f_8_2 + f_8_6)
                                    + 10496*(f_8_3 + f_8_5)
                                    - 4540*f_8_4)/567
        # 567 = 3 * 189 = 9 * 21 = 27 * 7
    return integral

###################
# ERROS - FUNÇÕES #
###################


def erroC0(funcao1, funcao2, dominio):
    '''
    Dadas as funções funcao1 e funcao2 sobre o mesmo domínio, calcula o erro
    no sentido do espaço C^0(dominio)
    '''
    dominio = any2vector(dominio)
    ftemp1 = func2vector(funcao1, dominio)
    ftemp2 = func2vector(funcao2, dominio)

    if isinstance(ftemp1, bool) or isinstance(ftemp2, bool):
        print('A imagem da primeira ou da segunda função\
              \ntem tamanho incompatível com o domínio')
        saida = False
    else:
        saida = max(np.absolute(ftemp1 - ftemp2))
    return saida


def erroL1(funcao1, funcao2, dominio):
    '''
    Dadas as funções funcao1 e funcao2 sobre o mesmo domínio, calcula o erro
    no sentido do espaço L^1(dominio)
    '''
    dominio = any2vector(dominio)
    ftemp1 = func2vector(funcao1, dominio)
    ftemp2 = func2vector(funcao2, dominio)

    if isinstance(ftemp1, bool) or isinstance(ftemp2, bool):
        print('A imagem da primeira ou da segunda função\
              \ntem tamanho incompatível com o domínio')
        saida = False
    else:
        vtemp = np.absolute(ftemp1 - ftemp2)
        delta = dominio[1] - dominio[0]
        saida = delta * (sum(vtemp) - 0.5*(vtemp[0] + vtemp[-1]))
    return saida


def erroL2(funcao1, funcao2, dominio):
    '''
    Dadas as funções funcao1 e funcao2 sobre o mesmo domínio, calcula o erro
    no sentido do espaço L^2(dominio)
    '''
    dominio = any2vector(dominio)
    ftemp1 = func2vector(funcao1, dominio)
    ftemp2 = func2vector(funcao2, dominio)

    if isinstance(ftemp1, bool) or isinstance(ftemp2, bool):
        print('A imagem da primeira ou da segunda função\
              \ntem tamanho incompatível com o domínio')
        saida = False
    else:
        vtemp = (ftemp1 - ftemp2)**2
        delta = dominio[1] - dominio[0]
        saida = delta * (sum(vtemp) - 0.5*(vtemp[0] + vtemp[-1]))
    return saida


def erroC1(funcao1, funcao2, dominio):
    '''
    Dadas as funções funcao1 e funcao2 sobre o mesmo domínio, calcula o erro
    no sentido do espaço C^1(dominio)
    '''
    dominio = any2vector(dominio)
    ftemp1 = func2vector(funcao1, dominio)
    ftemp2 = func2vector(funcao2, dominio)

    if isinstance(ftemp1, bool) or isinstance(ftemp2, bool):
        print('A imagem da primeira ou da segunda função\
              \ntem tamanho incompatível com o domínio')
        saida = False
    else:
        dftemp1 = funcao_df5p(funcao1, dominio)
        dftemp2 = funcao_df5p(funcao2, dominio)
        saida = (max(np.absolute(ftemp1 - ftemp2))
                 + max(np.absolute(dftemp1 - dftemp2)))
    return saida


def erroW11(funcao1, funcao2, dominio):
    '''
    Dadas as funções funcao1 e funcao2 sobre o mesmo domínio, calcula o erro
    no sentido do espaço W^{1, 1}(dominio)
    '''
    dominio = any2vector(dominio)
    ftemp1 = func2vector(funcao1, dominio)
    ftemp2 = func2vector(funcao2, dominio)
    dftemp1 = func2vector(funcao_df5p(funcao1, dominio), dominio)
    dftemp2 = func2vector(funcao_df5p(funcao2, dominio), dominio)

    if isinstance(ftemp1, bool) or isinstance(ftemp2, bool):
        print('A imagem da primeira ou da segunda função\
              \ntem tamanho incompatível com o domínio')
        saida = False
    else:
        vtemp = np.absolute(ftemp1 - ftemp2)
        dvtemp = np.absolute(dftemp1 - dftemp2)
        delta = dominio[1] - dominio[0]
        saida = delta * (sum(vtemp) - 0.5*(vtemp[0] + vtemp[-1])
                         + sum(dvtemp) - 0.5*(dvtemp[0] + dvtemp[-1]))
    return saida


def erroH1(funcao1, funcao2, dominio):
    '''
    Dadas as funções funcao1 e funcao2 sobre o mesmo domínio, calcula o erro
    no sentido do espaço L^2(dominio)
    '''
    dominio = any2vector(dominio)
    ftemp1 = func2vector(funcao1, dominio)
    ftemp2 = func2vector(funcao2, dominio)
    dftemp1 = func2vector(funcao_df5p(funcao1, dominio), dominio)
    dftemp2 = func2vector(funcao_df5p(funcao2, dominio), dominio)

    if isinstance(ftemp1, bool) or isinstance(ftemp2, bool):
        print('A imagem da primeira ou da segunda função\
              \ntem tamanho incompatível com o domínio')
        saida = False
    else:
        vtemp = (ftemp1 - ftemp2)**2
        dvtemp = (dftemp1 - dftemp2)**2
        delta = dominio[1] - dominio[0]
        saida = (delta * (sum(vtemp) - 0.5*(vtemp[0] + vtemp[-1])
                          + sum(dvtemp) - 0.5*(dvtemp[0] + dvtemp[-1])))**(0.5)
    return saida

####################################
# EQUAÇÕES DIFERENCIAIS ORDINARIAS #
####################################


def euler(dominio, y_0, funcao):
    '''
    Método de Euler - Resolve o Problema de Valor inicial

    y' = f(x,y), x in dominio
    y(dominio[0]) = y_0

    A funcao deve retornar vetores
    '''
    X = any2vector(dominio)

    delta = X[1] - X[0]
    n = len(X) - 1

    y_0 = any2vector(y_0)

    Y = np.empty([n+1, len(y_0)])

    Y[0] = y_0
    for j in range(0, n):
        Y[j+1] = Y[j] + delta * funcao(X[j], Y[j])

    return {"domain": X, "solution": Y}


def eulermod(dominio, y_0, funcao):
    '''
    Método de Euler modificado, Euler melhorado, ou Euler aprimorado

    A funcao deve retornar vetores
    '''
    X = any2vector(dominio)

    delta = X[1] - X[0]
    delta2 = 0.5 * delta
    n = len(X) - 1

    y_0 = any2vector(y_0)

    Y = np.empty([n + 1, len(y_0)])
    Y[0] = y_0
    for j in range(0, n):
        k1 = funcao(X[j], Y[j])
        k2 = funcao(X[j + 1], Y[j] + delta * k1)
        Y[j+1] = Y[j] + delta2 * (k1 + k2)

    return {"domain": X, "solution": Y}


def midpoint(dominio, y_0, funcao):
    '''
    Método do Ponto Médio

    A funcao deve retornar vetores
    '''
    X = any2vector(dominio)

    delta = X[1] - X[0]
    delta2 = 0.5 * delta
    n = len(X) - 1

    y_0 = any2vector(y_0)

    Y = np.empty([n + 1, len(y_0)])
    Y[0] = y_0
    for j in range(0, n):
        k1 = funcao(X[j], Y[j])
        k2 = funcao(X[j] + delta2, Y[j] + delta2 * k1)
        Y[j+1] = Y[j] + delta * k2

    return {"domain": X, "solution": Y}


def rk3(dominio, y_0, funcao):
    '''
    Método Runge-Kutta de terceira ordem

    A funcao deve retornar vetores
    '''
    X = any2vector(dominio)

    delta = X[1] - X[0]
    delta2 = 0.5 * delta
    delta6 = 0.5 * delta / 3
    n = len(X) - 1

    y_0 = any2vector(y_0)

    Y = np.empty([n + 1, len(y_0)])
    Y[0] = y_0
    for j in range(0, n):
        k1 = funcao(X[j], Y[j])
        k2 = funcao(X[j] + delta2, Y[j] + delta2 * k1)
        k3 = funcao(X[j] + delta, Y[j] - delta*k1 + 2*delta*k2)
        Y[j+1] = Y[j] + delta6 * (k1 + 4 * k2 + k3)

    return {"domain": X, "solution": Y}


def rk4classic(dominio, y_0, funcao):
    '''
    Método Runge-Kutta de quarta ordem clássico

    A funcao deve retornar vetores
    '''
    X = any2vector(dominio)

    delta = X[1] - X[0]
    delta2 = 0.5 * delta
    delta6 = delta / 6
    n = len(X) - 1

    y_0 = any2vector(y_0)

    Y = np.empty([n + 1, len(y_0)])
    Y[0] = y_0
    for j in range(0, n):
        k1 = funcao(X[j], Y[j])
        k2 = funcao(X[j] + delta2, Y[j] + delta2 * k1)
        k3 = funcao(X[j] + delta2, Y[j] + delta2 * k2)
        k4 = funcao(X[j + 1], Y[j] + delta * k3)
        Y[j+1] = Y[j] + delta6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return {"domain": X, "solution": Y}
