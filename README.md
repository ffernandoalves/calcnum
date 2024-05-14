# Calcnum

## Instalação

### Criando um Ambiente Virtual

É recomendado utilizar o ambiente virtual do Python, assim poderá ser possível utilizar várias versões da biblioteca. Caso queira instalar para todo o computador, basta ir para a próxima seção da instalação.

Abrar o terminal na pasta do seu projeto e execute nele o seguinte comando para criar o ambiente virtual:

```bash
$ python3 -m venv env
```

Para ativar, no terminal, execute:

1. Linux: `$ activate`
2. Windows: `$ env\Scripts\activate`

_Obs: Para desativar o ambiente virtual, execute: `$ deactivate`_

### Instalando a partir do Wheel

Para instalar via o wheel, veja como obter em <a href="#package">Package</a>, execute:.
```bash
$ python3 -m pip install wheel
```

## Forma de Usar

### Documentação da Biblioteca

Leia em [docs](./docs/index.md)

## Desenvolvimento

### Package

Bibliotecas necessárias para gerar o pacote:

```bash
$ python3 -m pip install -U setuptools
$ python3 -m pip install -U build
```

Comando para gerar o pacote:

```bash
$ python3 -m build
```

Leia mais em:

1. https://packaging.python.org/en/latest/tutorials/packaging-projects/
2. https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
3. https://setuptools.pypa.io/en/latest/userguide/declarative_config.html
4. https://packaging.python.org/pt-br/latest/discussions/setup-py-deprecated/
