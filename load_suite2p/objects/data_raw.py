from typing import Union

import h5py
import numpy as np


class DataRaw:
    def __init__(self, data, is_allen: bool = True):
        if is_allen:
            self.day = self.extract_data(data["day"], data)
            self.imaging = self.extract_data(data["imaging"], data)
            self.f = self.extract_data(data["f"], data)
            self.is_cell = self.extract_data(data["is_cell"], data)
            self.r_neu = self.extract_data(data["r_neu"], data)
            self.stim = self.extract_data(data["stim"], data)
            self.trig = self.extract_data(data["trig"], data)
        else:
            raise NotImplementedError(
                "Only loading for Allen data is implemented"
            )

    def dataset_to_dict(
        self,
        dataset: h5py._hl.dataset.Dataset,
        parent_container: h5py._hl.group.Group,
    ) -> dict:
        dict = {}
        for i in range(dataset.shape[1]):
            ref = dataset[0][i]
            dict = self.group_to_dict_recursive(parent_container[ref])
        return dict

    def group_to_dict_recursive(self, group: h5py._hl.group.Group) -> dict:
        dict = {}
        for key in group:
            if isinstance(group[key], h5py._hl.group.Group):
                dict[key] = self.group_to_dict_recursive(group[key])
            else:
                dict[key] = group[key][:]
        return dict

    def dataset_to_array(
        self,
        dataset: h5py._hl.dataset.Dataset,
        parent_container: Union[h5py._hl.group.Group, h5py.File],
    ) -> np.ndarray:
        array = np.zeros((dataset.shape[0], dataset.shape[1]), dtype=object)

        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                ref = dataset[i][j]
                array[i, j] = parent_container[ref][:]

        return array

    def extract_data(self, element, parent):
        if isinstance(element, h5py._hl.dataset.Dataset):
            if element.dtype == h5py.special_dtype(ref=h5py.Reference):
                test_ref = element[0][0]
                if isinstance(parent[test_ref], h5py._hl.group.Group):
                    return self.dataset_to_dict(element, parent)
                else:
                    return self.dataset_to_array(element, parent)
            else:
                return element[:]
        elif isinstance(element, h5py._hl.group.Group):
            dict = {}
            for key in element:
                dict[key] = self.extract_data(element[key], element)
            return dict
        else:
            raise TypeError("Element is neither a dataset nor a group")
