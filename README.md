# Visión por computador (2022-23)
Repositorio de la asignatura de Visión por computador de la Universidad de Zaragoza (2022-23).

## Preparación del entorno virtual
1. (Opcional, si no se tiene ya) Descargar [Python](https://www.python.org/downloads/).

2. (Opcional, si no se tiene ya) Comprobar si se tiene instalado `pip`.
```console
python -m pip --version
```
```console
pip --version
```

3. (Opcional, si no se tiene ya) Instalar `virtualenv` y comprobar que se ha instalado.
```console
python -m pip install virtualenv && python -m virtualenv --version
```
```console
pip install virtualenv && virtualenv --version
```

4. Elige tu carpeta destino y crea tu entorno virtual (no necesariamente tiene porque llamarse "vision").
```console
python -m virtualenv vision
```
```console
virtualenv vision
```

5. Inicia tu entorno virtual (suponiendo que tu entorno virtual se llama "vision").
```console
./vision/Scripts/activate
```

6. Salir del entorno virtual.
```
deactivate
```

## Preparación de OpenCV
Dentro del entorno virtual, instala las librerias necesarias para OpenCV:
```console
python -m pip install numpy matplotlib opencv-python
``` 
```console
pip install numpy matplotlib opencv-python
``` 

Ya tienes todo lo necesario para utilizar OpenCV, de momento.
