import os

import h5py
import numpy as np
import pytest

from load_suite2p.objects.data_raw import DataRaw


@pytest.fixture
def array_for_simple_dataset():
    yield np.arange(1, 5)


@pytest.fixture
def array_for_dataset_in_group():
    yield np.arange(1, 10)


@pytest.fixture
def array_for_dataset_in_subgroup():
    yield np.arange(1, 20)


@pytest.fixture
def array_for_dataset_with_references_to_dataset():
    yield np.arange(1, 30)


@pytest.fixture
def array_for_dataset_with_references_to_group_with_subsgourp():
    yield np.arange(1, 40)


@pytest.fixture
def get_path_to_hdf5_file(tmp_path_factory):
    yield tmp_path_factory.mktemp("data") / "mytestfile.hdf5"


@pytest.fixture
def create_mock_hdf5_data(
    array_for_simple_dataset,
    array_for_dataset_in_group,
    array_for_dataset_in_subgroup,
    array_for_dataset_with_references_to_dataset,
    array_for_dataset_with_references_to_group_with_subsgourp,
    get_path_to_hdf5_file,
):
    with h5py.File(get_path_to_hdf5_file, "w") as f:
        # create a file and add a simple dataset
        f.create_dataset("array0", data=array_for_simple_dataset, dtype="i")

        # create a file and add a group with a dataset inside
        grp = f.create_group("mygroup")
        grp.create_dataset(
            "array1", data=array_for_dataset_in_group, dtype="f"
        )

        # create group with subgroup and a dataset
        subgroup = grp.create_group("subgroup")
        subgroup.create_dataset(
            "array2", data=array_for_dataset_in_subgroup, dtype="f"
        )

        # create a dataset with references of dataset
        dataset_to_be_referenced = f.create_dataset(
            "array3",
            data=array_for_dataset_with_references_to_dataset,
            dtype="f",
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
        subgroup2.create_dataset(
            "array4",
            data=array_for_dataset_with_references_to_group_with_subsgourp,
            dtype="f",
        )
        ref2 = group_to_be_referenced.ref
        ref_array2 = [[ref2, ref2], [ref2, ref2]]
        f.create_dataset(
            "ref_dataset2",
            data=ref_array2,
            dtype=h5py.special_dtype(ref=h5py.Reference),
        )

    yield None


@pytest.fixture(autouse=True)
def clean_up(get_path_to_hdf5_file):
    yield

    os.remove(get_path_to_hdf5_file)


def test_unpack_of_simple_dataset(
    create_mock_hdf5_data, array_for_simple_dataset, get_path_to_hdf5_file
):
    with h5py.File(get_path_to_hdf5_file, "r") as f:
        assert np.all(
            DataRaw._unpack_data(f["array0"], f) == array_for_simple_dataset
        )


def test_unpack_of_dataset_in_group(
    create_mock_hdf5_data, array_for_dataset_in_group, get_path_to_hdf5_file
):
    with h5py.File(get_path_to_hdf5_file, "r") as f:
        assert np.all(
            DataRaw._unpack_data(f["mygroup"]["array1"], f)
            == array_for_dataset_in_group
        )


def test_unpack_of_dataset_in_subgroup(
    create_mock_hdf5_data, array_for_dataset_in_subgroup, get_path_to_hdf5_file
):
    with h5py.File(get_path_to_hdf5_file, "r") as f:
        assert np.all(
            DataRaw._unpack_data(f["mygroup"]["subgroup"]["array2"], f)
            == array_for_dataset_in_subgroup
        )


def test_unpack_of_dataset_with_references_to_dataset(
    create_mock_hdf5_data,
    array_for_dataset_with_references_to_dataset,
    get_path_to_hdf5_file,
):
    with h5py.File(get_path_to_hdf5_file, "r") as f:
        assert np.all(
            DataRaw._unpack_data(f["ref_dataset"], f)[0][0]
            == array_for_dataset_with_references_to_dataset
        )


def test_unpack_of_dataset_with_references_to_group_with_subgroup(
    create_mock_hdf5_data,
    array_for_dataset_with_references_to_group_with_subsgourp,
    get_path_to_hdf5_file,
):
    with h5py.File(get_path_to_hdf5_file, "r") as f:
        assert np.all(
            DataRaw._unpack_data(f["ref_dataset2"], f)[0][0]["subgroup2"][
                "array4"
            ]
            == array_for_dataset_with_references_to_group_with_subsgourp
        )
