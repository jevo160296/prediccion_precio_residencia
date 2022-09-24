import pathlib

import joblib
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List


class Paso(ABC):
    def __init__(self, nombre, paso_anterior=None):
        self._nombre = nombre
        self._paso_anterior = paso_anterior
        self._respuesta = None
        self._executed = False

    @abstractmethod
    def _run(self, **kwargs) -> dict:
        print(f'Ejecutando {self._nombre}')
        return {}

    def run(self, force_execution=False, **kwargs):
        if self._executed and not force_execution:
            return self._respuesta
        elif self._paso_anterior is not None:
            self._respuesta = self._run(**self._paso_anterior.run(**kwargs))
        else:
            self._respuesta = self._run(**kwargs)
        self._executed = True
        return self._respuesta


class Proceso:
    def __init__(self, cached=False, cache_path=None):
        self._cached = cached
        self._cache_path = cache_path

    def save_cache(self):
        if self._cached:
            self._cache_path.parent.mkdir(exist_ok=True, parents=True)
            joblib.dump(self, self._cache_path)

    def clean_cache(self):
        if self._cache_path is not None and self._cache_path.exists():
            print("Limpiando cache")
            self._cache_path.unlink()
            self._cached = False

    @classmethod
    def from_cache(cls, def_process, cached=False, cache_path=None):
        if cached and cache_path.exists():
            r = joblib.load(cache_path)
            return r
        else:
            return def_process
