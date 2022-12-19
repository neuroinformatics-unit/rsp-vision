from pathlib import Path

from load_suite2p.objects.parsers2p.parser2pRSP import Parser2pRSP

config = {
    "parser": "Parser2pRSP",
    "paths": {
        "imaging": "test_data/",
        "allen-dff": "test_data/allen_dff/",
        "serial2p": "test_data/serial2p/",
        "stimulus-ai-schedule": "test_data/stimulus_ai_schedule/",
    },
}


def test_parser_2pRSP():
    parser = Parser2pRSP("CX_1111783_hR_RSPg_monitor_front", {})
    assert parser.info["mouse_line"] == "CX"
    assert parser.info["mouse_id"] == "1111783"
    assert parser.info["hemisphere"] == "hR"
    assert parser.info["brain_region"] == "RSPg"
    assert parser.info["monitor_position"] == "monitor_front"


def test_get_parent_folder_name():
    parser = Parser2pRSP("CX_1111783_hR_RSPg_monitor_front", {})
    assert parser._get_parent_folder_name() == "CX_1111783"


def test_get_path():
    parser = Parser2pRSP("CX_1111783_hR_RSPg_monitor_front", config)
    assert parser.get_path_to_experimental_folder() == Path(
        "test_data/CX_1111783/CX_1111783_hR_RSPg_monitor_front"
    )


def test_get_path_to_allen_dff_file():
    parser = Parser2pRSP("CX_1111783_hR_RSPg_monitor_front", config)
    assert parser.get_path_to_allen_dff_file() == Path(
        "test_data/allen_dff/"
        + "CX_1111783_hR_RSPg_monitor_front_sf_tf_allen_dff.mat"
    )


def test_get_path_to_serial2p():
    parser = Parser2pRSP("CX_1111783_hR_RSPg_monitor_front", config)
    assert parser.get_path_to_serial2p() == Path(
        "test_data/serial2p/CT_CX_1111783/"
    )
