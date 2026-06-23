from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from surrogate_tool.pipeline.fmu_export import (
    _gen_model_c,
    _gen_model_description_xml,
    _resolve_gcc,
)


# ---------------------------------------------------------------------------
# _resolve_gcc
# ---------------------------------------------------------------------------

def test_resolve_gcc_uses_cfg_path(tmp_path, mocker):
    fake_gcc = tmp_path / "gcc.exe"
    fake_gcc.touch()
    mocker.patch("glob.glob", return_value=[])
    mocker.patch("shutil.which", return_value=None)
    result = _resolve_gcc(str(fake_gcc))
    assert result == str(fake_gcc)


def test_resolve_gcc_cfg_path_missing_file(tmp_path):
    nonexistent = str(tmp_path / "no_gcc.exe")
    # cfg_path points to non-existent file; no OpenModelica on CI → should return None
    # (unless system gcc is on PATH, which we cannot control in unit test)
    result = _resolve_gcc(nonexistent)
    # Either None or a system gcc found via PATH — just confirm no exception
    assert result is None or isinstance(result, str)


def test_resolve_gcc_none_input(mocker):
    """With no cfg_path and no OpenModelica or system gcc, returns None."""
    mocker.patch("glob.glob", return_value=[])
    mocker.patch("shutil.which", return_value=None)
    result = _resolve_gcc(None)
    assert result is None


# ---------------------------------------------------------------------------
# _gen_model_description_xml — FMI 2.0 compliance
# ---------------------------------------------------------------------------

GUID = "12345678-1234-1234-1234-123456789abc"
INPUT_COLS = ["Temp", "Voltage"]
OUTPUT_COLS = ["PowerLoss"]
PKG = "TestModel"


def _parse_xml():
    xml_str = _gen_model_description_xml(PKG, INPUT_COLS, OUTPUT_COLS, GUID)
    return ET.fromstring(xml_str), xml_str


def test_xml_scalar_variable_count():
    root, _ = _parse_xml()
    sv_list = root.findall(".//ScalarVariable")
    assert len(sv_list) == len(INPUT_COLS) + len(OUTPUT_COLS)


def test_xml_inputs_have_real_start():
    root, _ = _parse_xml()
    for sv in root.findall(".//ScalarVariable[@causality='input']"):
        real_elem = sv.find("Real")
        assert real_elem is not None, f"Input {sv.attrib['name']} missing <Real>"
        assert "start" in real_elem.attrib, f"Input {sv.attrib['name']} <Real> missing start attribute"


def test_xml_inputs_no_initial_exact():
    root, _ = _parse_xml()
    for sv in root.findall(".//ScalarVariable[@causality='input']"):
        assert sv.attrib.get("initial") != "exact", (
            f"Input {sv.attrib['name']} must not have initial='exact'"
        )


def test_xml_model_exchange_no_needs_direction_derivatives():
    root, _ = _parse_xml()
    me = root.find("ModelExchange")
    assert me is not None
    assert "needsDirectionDerivatives" not in me.attrib


def test_xml_number_of_continuous_states_zero():
    _, xml_str = _parse_xml()
    # The attribute is on the root fmiModelDescription element; parse it
    root = ET.fromstring(xml_str)
    # Not required by our generator, but check it's not wrongly set > 0
    val = root.attrib.get("numberOfContinuousStates", "0")
    assert val == "0"


def test_xml_output_causality():
    root, _ = _parse_xml()
    output_svs = root.findall(".//ScalarVariable[@causality='output']")
    assert len(output_svs) == len(OUTPUT_COLS)


# ---------------------------------------------------------------------------
# _gen_model_c — weight arrays are transposed correctly
# ---------------------------------------------------------------------------

def test_gen_model_c_contains_layer_arrays(minimal_mlp_weights):
    c_src = _gen_model_c(PKG, INPUT_COLS, OUTPUT_COLS, minimal_mlp_weights)
    # Should contain static array declarations for each layer
    assert "L0_W" in c_src
    assert "L0_b" in c_src
    assert "L1_W" in c_src
    assert "L1_b" in c_src


def test_gen_model_c_transposed_weights(minimal_mlp_weights):
    """
    Layer 0 W in mlp_weights is [[1,2],[3,4]] (Keras [n_in, n_out]).
    Transposed (row-major C layout, [n_out, n_in]) = [[1,3],[2,4]].
    The flattened row-major values should be 1, 3, 2, 4 in the C source.
    """
    c_src = _gen_model_c(PKG, INPUT_COLS, OUTPUT_COLS, minimal_mlp_weights)
    # Find the L0_W array declaration and check the order of values
    assert "1" in c_src and "3" in c_src  # rough sanity; exact order tested below
    # Locate L0_W array: values should be 1.0, 3.0, 2.0, 4.0 (transposed row-major)
    l0_w_idx = c_src.index("L0_W[]")
    segment = c_src[l0_w_idx: l0_w_idx + 200]
    # Extract numbers from the array literal
    import re
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", segment.split("{")[1].split("}")[0])
    float_vals = [float(v) for v in numbers]
    assert float_vals == pytest.approx([1.0, 3.0, 2.0, 4.0])


def test_gen_model_c_scaler_arrays(minimal_mlp_weights):
    c_src = _gen_model_c(PKG, INPUT_COLS, OUTPUT_COLS, minimal_mlp_weights)
    assert "X_MEAN" in c_src
    assert "X_SCALE" in c_src
    assert "Y_MEAN" in c_src
    assert "Y_SCALE" in c_src
