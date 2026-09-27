import importlib

import pytest


ht = importlib.import_module("branch_extended_AD.HashTensor")


def test_branch_state_restores_after_nested_contexts():
    inactive = ht._branch_state.get()
    with ht._branch_mode("record", atol=0.25, rtol=0.1, tol_mode="input_scaled") as trace:
        recording = ht._branch_state.get()
        assert recording.mode == "record"
        assert recording.trace is trace
        assert recording.atol == 0.25
        assert recording.rtol == 0.1

        path = [ht._TraceNode("max", [0])]
        with ht._branch_mode("replay", replay_path=path):
            assert ht._branch_state.get().mode == "replay"
        assert ht._branch_state.get() is recording

    assert ht._branch_state.get() is inactive


def test_branch_state_restores_after_exception():
    inactive = ht._branch_state.get()
    with pytest.raises(RuntimeError):
        with ht._branch_mode("record"):
            raise RuntimeError("stop")
    assert ht._branch_state.get() is inactive
