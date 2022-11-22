from load_suite2p import folder_naming_specs

folder_test_list = [
    "AS_1111877_hL_V1_monitor_front",
    "AS_130_3_hL_V1_monitor_right",
    "BY_317_2_hL_RSPd_monitor_left",
    "BY_IAA_1117275_hL_RSPg_monitor_front",
    "CX_79_2_hL_RSPd_monitor_right",
    "CX_102_2_hL_RSPd_FOV3c_monitor_front",
    "CX_1111923_hL_V1_monitor_front-right",
    "SG_1118210_hR_RSPd_cre-off_monitor_front",
]


def test_FileNamingSpecs_constructor():
    for folder in folder_test_list:
        folder_naming_specs.FolderNamingSpecs(folder)


def test_FileNamingSpecs_constructor_fails():
    control_exception = False
    try:
        folder_naming_specs.FolderNamingSpecs(
            "AS_1111877_hL_V1_monitor_front_wrong"
        )
    except FileNotFoundError:
        control_exception = True
        pass
    if control_exception is False:
        raise AssertionError("FileNotFoundError not raised, filters failed")
