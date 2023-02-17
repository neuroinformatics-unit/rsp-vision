import logging
from typing import Union

import h5py
import numpy as np


class DataRaw:
    """Class to load and contain the raw data.
    It can load data from Allen or from
    a list of Paths. Only the Allen case is implemented so far.
    """

    def __init__(self, data: dict, is_allen: bool = True):
        if is_allen:
            logging.info("Loading Allen data, starting to unpack...")

            self.day = self._unpack_data(data["day"], data)
            logging.info("Unpacked day")

            self.imaging = self._unpack_data(data["imaging"], data)
            logging.info("Unpacked imaging")

            self.frames = self._unpack_data(data["f"], data)
            logging.info("Unpacked f")

            self.is_cell = self._unpack_data(data["is_cell"], data)
            logging.info("Unpacked is_cell")

            self.neuropil_coeficient = self._unpack_data(data["r_neu"], data)
            logging.info("Unpacked r_neu")

            self.stim = self._unpack_data(data["stim"], data)
            logging.info("Unpacked stim")

            self.trig = self._unpack_data(data["trig"], data)
            logging.info("Unpacked trig")
        else:
            self.day = data["day"]
            self.imaging = data["imaging"]
            self.frames = data["f"]
            self.is_cell = data["is_cell"]
            self.neuropil_coeficient = data["r_neu"]
            self.stim = data["stim"]
            self.trig = data["trig"]

    def __repr__(self) -> str:
        return f"DataRaw(day={self.day}, imaging={self.imaging}, \
            f={self.frames}, is_cell={self.is_cell}, \
            r_neu={self.neuropil_coeficient}, stim={self.stim}, \
            trig={self.trig})"

    @classmethod
    def _unpack_data(
        cls,
        element: Union[h5py._hl.dataset.Dataset, h5py._hl.group.Group],
        parent: Union[h5py.File, h5py._hl.group.Group],
    ) -> Union[np.ndarray, dict]:
        """This method unpack a complex MATLAB datastructure and returns a
        nested dictionary or numpy array. Only the relevant subset (Dataset
        and Groups) of the possible datastructures is implemented.
        Datasets can be mapped to arrays. Groups can be mapped to
        dictionaries, and each entry can be a Dataset or another Group.
        An array might contain numbers or point to other Arrays or Groups
        through References.
        References are a HDF5 type that can point either to an array or
        to a group.
        They need to be resolved in order to get the data. They are resolved
        by calling the methods ref_dataset_to_array.
        If element is a Group, its content is unpacked recursively.


        Example of folder structure:
        . (root)
        ├── dataset_01 (contains numbers, -> array)
        ├── dataset_02 (contains references to datasets, -> array)
        ├── dataset_03 (contains references to groups, -> array of dict)
        ├── group_01 (contains datasets and groups, never references
                    -> dict of arrays and dicts)

        A specific example:
        `data["day"]` is a group containing datasets or groups. It is
        mappable to a dictionary.
        When `unpack_data()` is called on `data["day"]`, `isinstance(element,
        h5py._hl.group.Group)` will be true and the method will call itself
        recursively on each element in the group, unpacking datasets and groups
        until it reaches the bottom of the tree.
        This is one of the most complicated matlab `struct` to unpack,
        together with `data["stim"]`.

        Args:
            element Union[h5py._hl.dataset.Dataset, h5py._hl.group.Group]:
                is either a h5py Group or Dataset.
                It is what we want to unpack.
            parent Union[h5py.File, h5py._hl.group.Group]:
                is the object that contains the element.
                It is used to resolve references.

        Returns:
            Union[np.ndarray, dict]:
                is either a numpy array or a nested dictionary.
        """
        if isinstance(element, h5py._hl.dataset.Dataset):
            if element.dtype == h5py.special_dtype(ref=h5py.Reference):
                return cls._ref_dataset_to_array(element, parent)
            else:
                return np.squeeze(element[:])
        elif isinstance(element, h5py._hl.group.Group):
            dict = {}
            for key in element:
                dict[key] = cls._unpack_data(element[key], element)
            return dict
        else:
            return None

    @classmethod
    def _ref_dataset_to_array(
        cls,
        dataset: h5py._hl.dataset.Dataset,
        parent: Union[h5py._hl.group.Group, h5py.File],
    ) -> np.ndarray:
        """Takes a Dataset that contains references to other Datasets or
        Groups and resolves its content.

        Args:
            dataset (h5py._hl.dataset.Dataset):
                HDF5 Dataset containing references
            parent_container (Union[h5py._hl.group.Group, h5py.File]):
                is the object that contains the element.
                It is used to resolve references.

        Returns:
            np.ndarray: an array of numbers or an array of dictionaries
        """
        array = np.zeros((dataset.shape[0], dataset.shape[1]), dtype=object)

        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                ref = dataset[i][j]
                if isinstance(parent[ref], h5py._hl.group.Group):
                    array[i, j] = cls._group_to_dict_recursive(parent[ref])
                else:
                    array[i, j] = np.squeeze(parent[ref][:])

        return np.squeeze(array)

    @classmethod
    def _group_to_dict_recursive(cls, group: h5py._hl.group.Group) -> dict:
        """Takes a Group and resolves its content. If the Group contains
        other Groups, it calls itself recursively.
        It assumes there are no more References.

        Args:
            group (h5py._hl.group.Group):
                HDF5 Group containing references

        Returns:
            dict: the resolved dictionary
        """
        dict = {}
        for key in group:
            if isinstance(group[key], h5py._hl.group.Group):
                dict[key] = cls._group_to_dict_recursive(group[key])
            else:
                dict[key] = np.squeeze(group[key][:])
        return dict
