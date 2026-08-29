from __future__ import annotations

from imu_benchmark.splits import ParticipantStats, propose_assignments


def _stats(dataset: str, participant: str, weight: int) -> ParticipantStats:
    return ParticipantStats(
        dataset_id=dataset,
        participant_id=participant,
        sequences=weight,
        fall_sequences=weight,
        events=weight,
        adl_rows=weight * 100,
        body_locations=("waist",),
    )


def test_sticky_split_keeps_existing_and_deterministically_assigns_new_participants() -> None:
    stats = {
        ("dataset", f"p{index}"): _stats("dataset", f"p{index}", index + 1)
        for index in range(10)
    }
    existing = {("dataset", f"p{index}"): index for index in range(5)}
    first, new_keys = propose_assignments(stats, existing)
    second, repeated_new_keys = propose_assignments(stats, existing)
    assert first == second
    assert new_keys == repeated_new_keys
    assert all(first[key] == fold for key, fold in existing.items())
    assert set(first.values()) == set(range(5))


def test_sticky_split_rejects_assignments_for_absent_participants() -> None:
    stats = {("dataset", "p0"): _stats("dataset", "p0", 1)}
    try:
        propose_assignments(stats, {("dataset", "missing"): 0})
    except ValueError as error:
        assert "absent from data" in str(error)
    else:
        raise AssertionError("Expected absent participant assignment to fail")
