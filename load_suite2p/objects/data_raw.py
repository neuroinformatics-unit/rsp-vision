from typing import Union

import h5py
import numpy as np


class DataRaw:
    """Class to load and contain the raw data.
    It can load data from Allen or from
    a list of Paths. Only the Allen case is implemented so far.
    """

    def __init__(self, data, is_allen: bool = True):
        if is_allen:
            self.day = self.unpack_data(data["day"], data)
            self.imaging = self.unpack_data(data["imaging"], data)
            self.f = self.unpack_data(data["f"], data)
            self.is_cell = self.unpack_data(data["is_cell"], data)
            self.r_neu = self.unpack_data(data["r_neu"], data)
            self.stim = self.unpack_data(data["stim"], data)
            self.trig = self.unpack_data(data["trig"], data)
        else:
            raise NotImplementedError(
                "Only loading for Allen data is implemented"
            )

    def ref_dataset_to_dict(
        self,
        dataset: h5py._hl.dataset.Dataset,
        parent: h5py._hl.group.Group,
    ) -> dict:
        """Takes a Dataset that contains references to Groups and
        resolves its content.

        Args:
            dataset (h5py._hl.dataset.Dataset):
                HDF5 Dataset containing references
            parent_container (h5py._hl.group.Group):
                is the object that contains the element.
                It is used to resolve references.

        Returns:
            dict: the resolved dictionary
        """
        dict = {}
        for i in range(dataset.shape[1]):
            ref = dataset[0][i]
            dict = self.group_to_dict_recursive(parent[ref])
        return dict

    def group_to_dict_recursive(self, group: h5py._hl.group.Group) -> dict:
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
                dict[key] = self.group_to_dict_recursive(group[key])
            else:
                dict[key] = group[key][:]
        return dict

    def ref_dataset_to_array(
        self,
        dataset: h5py._hl.dataset.Dataset,
        parent: Union[h5py._hl.group.Group, h5py.File],
    ) -> np.ndarray:
        """Takes a Dataset that contains references to numbers
        and resolves its content.

        Args:
            dataset (h5py._hl.dataset.Dataset):
                HDF5 Dataset containing references
            parent_container (Union[h5py._hl.group.Group, h5py.File]):
                is the object that contains the element.
                It is used to resolve references.

        Returns:
            np.ndarray: the resolved array
        """
        array = np.zeros((dataset.shape[0], dataset.shape[1]), dtype=object)

        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                ref = dataset[i][j]
                array[i, j] = parent[ref][:]

        return array

    def unpack_data(
        self,
        element: Union[h5py._hl.dataset.Dataset, h5py._hl.group.Group],
        parent: Union[h5py.File, h5py._hl.group.Group],
    ) -> Union[np.ndarray, dict]:
        """This method unpack a complex MATLAB datastructure and returns a
        nested dictionary or numpy array. Only the relevant subset (Dataset
        and Groups) of the possible datastructures is implemented.
        Datasets can be mapped to arrays. Groups can be mapped to
        dictionaires, and each entry can be a Dataset or another Group.
        An array might contain numbers or point to other Arrays or Groups
        through References.
        References are a HDF5 type that can point either to an array or
        to a group.
        They need to be resolved in order to get the data. They are resolved
        by calling the methods ref_dataset_to_dict or ref_dataset_to_array.
        If element is a Group, its content is unpacked recursively.

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
                test_ref = element[0][0]
                if isinstance(parent[test_ref], h5py._hl.group.Group):
                    return self.ref_dataset_to_dict(element, parent)
                else:
                    return self.ref_dataset_to_array(element, parent)
            else:
                return element[:]
        elif isinstance(element, h5py._hl.group.Group):
            dict = {}
            for key in element:
                dict[key] = self.unpack_data(element[key], element)
            return dict
