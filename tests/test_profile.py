import json

import pytest

from guidon.profile import collective_records, summarize


def test_tuple_collective_payloads_include_all_members_once():
    hlo = """
    %a = (bf16[768,2304]{1,0:T(8,128)}, f32[]{:T(128)}) all-reduce(%x, %y), replica_groups=[1,16]<=[16], use_global_device_ids=true
    %b = bf16[768,2304] get-tuple-element(%a), index=0
    %c = f32[4,8] all-gather(%z), replica_groups={{0,1,2,3}}, use_global_device_ids=true
    %async = (f32[4], f32[4], u32[]) all-reduce-start(%t), replica_groups=[1,16]<=[16], use_global_device_ids=true
    %result = f32[4] all-reduce-done(%async)
    """
    records = collective_records(hlo)
    assert len(records) == 3
    assert records[0]["output_bytes"] == 768 * 2304 * 2 + 4
    assert records[0]["tuple_members"] == 2
    assert records[1]["output_bytes"] == 4 * 8 * 4
    assert records[2]["output_bytes"] == 4 * 4


def test_incomplete_profile_cannot_be_promoted(tmp_path):
    (tmp_path / "completion-000040-worker0.json").write_text(
        json.dumps({"purpose": "profile"})
    )
    (tmp_path / "updates-worker0.jsonl").write_text(
        "\n".join(json.dumps({"update": i}) for i in range(1, 41))
    )
    with pytest.raises(ValueError, match="200 synchronized steady"):
        summarize(tmp_path, tmp_path / "summary.json")
    assert not (tmp_path / "summary.json").exists()
