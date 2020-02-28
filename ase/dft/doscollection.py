from abc import ABCMeta
from collections.abc import Sequence
from functools import singledispatch
from typing import Iterable, Union

from ase.dft.dosdata import DOSData, RawDOSData, GridDOSData


class DOSCollection(Sequence, metaclass=ABCMeta):
    """Abstract base class for a collection of DOSData objects"""
    def __init__(self, dos_series: Iterable[DOSData]) -> None:
        self._data = list(dos_series)

    def __getitem__(self, key: int):
        if not isinstance(key, int):
            raise TypeError("index in DOSCollection must be an integer")
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __add__(self,
                other: Union['DOSCollection', DOSData]) -> 'DOSCollection':
        """Join entries between two DOSCollection objects of the same type

        It is also possible to add a single DOSData object without wrapping it
        in a new collection: i.e.

          DOSCollection([dosdata1]) + DOSCollection([dosdata2])

        or

          DOSCollection([dosdata1]) + dosdata2

        will return

          DOSCollection([dosdata1, dosdata2])

        """
        return _add_to_collection(other, self)


@singledispatch
def _add_to_collection(other: DOSData,
                       collection: DOSCollection) -> DOSCollection:
    if isinstance(other, type(collection)):
        return type(collection)(list(collection) + list(other))
    else:
        raise TypeError("Only DOSCollection objects of the same type may "
                        "be joined with '+'.")


@_add_to_collection.register(DOSData)
def _(other: DOSData, collection: DOSCollection) -> DOSCollection:
    """Return a new DOSCollection with an additional DOSData item"""
    if isinstance(other, DOSData):
        return type(collection)(list(collection) + [other])


class RawDOSCollection(DOSCollection):
    def __init__(self, dos_series: Iterable[RawDOSData]) -> None:
        super().__init__(dos_series)
        for dos_data in self:
            if not isinstance(dos_data, RawDOSData):
                raise TypeError("RawDOSCollection can only store "
                                "RawDOSData objects.")

class GridDOSCollection(DOSCollection):
    def __init__(self, dos_series: Iterable[GridDOSData]) -> None:
        dos_list = list(dos_series)
        self._energies = dos_list[0].get_energies()

        for dos_data in dos_list:
            if not isinstance(dos_data, GridDOSData):
                raise TypeError("GridDOSCollection can only store "
                                "GridDOSData objects.")
            if not np.allclose(dos_data.get_energies(), self._energies):
                raise ValueError("All GridDOSData objects in GridDOSCollection"
                                 " must have the same energy axis.")
