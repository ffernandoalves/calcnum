import src.calcnum as cn


def tfuncao(x):
    return x

def test_particao():
    """
    A função `particao` retorna N+1 elementos
    """
    a = 0
    b = 1
    N = 10
    result = cn.particao(a, b, N)
    assert len(result) == N+1


def test_imagem():
    a = 0
    b = 1
    N = 10
    dominio = cn.particao(a, b, N)
    result = cn.imagem(tfuncao, dominio)
    assert len(result) == len(dominio)


def test_grafico():
    # cn.grafico()
    pass


def test_erro_abs():
    pass


def test_erro_rel():
    pass


def test_any2vector():
    pass


def test_func2vector():
    pass


def test_bisseccao():
    pass


def test_pontofixo():
    pass


def test_newtonraphson():
    pass


def test_regulafalsi():
    pass


def test_secantes():
    pass


def test_dfmais():
    pass


def test_dfmenos():
    pass


def test_dfcentro():
    pass


def test_d2f():
    pass


def test_df5p():
    pass


def test_d2f5p():
    pass


def test_d3f5p():
    pass


def test_d4f5p():
    pass


def test_dfmais3p():
    pass


def test_d2fmais3p():
    pass


def test_dfmais4p():
    pass


def test_d2fmais4p():
    pass


def test_d3fmais4p():
    pass


def test_dfmais5p():
    pass


def test_dfmenos3p():
    pass


def test_d2fmenos3p():
    pass


def test_dfmenos4p():
    pass


def test_d2fmenos4p():
    pass


def test_d3fmenos4p():
    pass


def test_dfmenos5p():
    pass


def test_funcao_df3p():
    pass


def test_funcao_df5p():
    pass


def test_intmidpoint():
    pass


def test_trapezio():
    pass


def test_simpson():
    pass


def test_simpson38():
    pass


def test_boole():
    pass


def test_int6pclosed():
    pass


def test_int7pclosed():
    pass


def test_int8pclosed():
    pass


def test_int9pclosed():
    pass


def test_erroC0():
    pass


def test_erroL1():
    pass


def test_erroL2():
    pass


def test_erroC1():
    pass


def test_erroW11():
    pass


def test_erroH1():
    pass


def test_euler():
    pass


def test_eulermod():
    pass


def test_midpoint():
    pass


def test_rk3():
    pass


def test_rk4classic():
    pass

