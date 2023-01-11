import h5py
import numpy as np

from load_suite2p.objects.data_raw import DataRaw

array0 = np.array([1, 2, 3, 4, 5])
array1 = np.array([3, 4, 5, 6, 7])
array2 = np.array([5, 6, 7, 8, 9])
array3 = np.array([7, 8, 9, 10, 11])
array4 = np.array([9, 10, 11, 12, 13])


def create_mock_hdf5_data():
    with h5py.File("mytestfile.hdf5", "w") as f:
        # create a file and add a simple dataset
        f.create_dataset("array0", data=array0, dtype="i")

        # create a file and add a group with a dataset inside
        grp = f.create_group("mygroup")
        grp.create_dataset("array1", data=array1, dtype="f")

        # create group with subgroup and a dataset
        subgroup = grp.create_group("subgroup")
        subgroup.create_dataset("array2", data=array2, dtype="f")

        # create a dataset with references of dataset
        dataset_to_be_referenced = f.create_dataset(
            "array3", data=array3, dtype="f"
        )
        ref = dataset_to_be_referenced.ref
        ref_array = [[ref, ref], [ref, ref]]
        f.create_dataset(
            "ref_dataset",
            data=ref_array,
            dtype=h5py.special_dtype(ref=h5py.Reference),
        )

        # create a dataset with references of group with subgroup
        group_to_be_referenced = f.create_group("#ref_group#")
        subgroup2 = group_to_be_referenced.create_group("subgroup2")
        subgroup2.create_dataset("array4", data=array4, dtype="f")
        ref2 = group_to_be_referenced.ref
        ref_array2 = [[ref2, ref2], [ref2, ref2]]
        f.create_dataset(
            "ref_dataset2",
            data=ref_array2,
            dtype=h5py.special_dtype(ref=h5py.Reference),
        )


def test_unpack_of_simple_dataset():
    create_mock_hdf5_data()
    with h5py.File("mytestfile.hdf5", "r") as f:
        assert np.all(DataRaw.unpack_data(f["array0"], f) == array0)


def test_unpack_of_dataset_in_group():
    create_mock_hdf5_data()
    with h5py.File("mytestfile.hdf5", "r") as f:
        assert np.all(DataRaw.unpack_data(f["mygroup"]["array1"], f) == array1)


def test_unpack_of_dataset_in_subgroup():
    create_mock_hdf5_data()
    with h5py.File("mytestfile.hdf5", "r") as f:
        assert np.all(
            DataRaw.unpack_data(f["mygroup"]["subgroup"]["array2"], f)
            == array2
        )


def test_unpack_of_dataset_with_references_to_dataset():
    create_mock_hdf5_data()
    with h5py.File("mytestfile.hdf5", "r") as f:
        assert np.all(DataRaw.unpack_data(f["ref_dataset"], f)[0][0] == array3)


def test_unpack_of_dataset_with_references_to_group_with_subgroup():
    create_mock_hdf5_data()
    with h5py.File("mytestfile.hdf5", "r") as f:
        assert np.all(
            DataRaw.unpack_data(f["ref_dataset2"], f)[0][0]["subgroup2"][
                "array4"
            ]
            == array4
        )
