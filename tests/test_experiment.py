from config import RUNS_DIR
from pipeline.experiment import (
    ExperimentConfig,
    RunMetadata,
    create_run_id,
    mark_run_finished,
    save_experiment_config,
    save_run_metadata,
)


def test_save_experiment_config_and_metadata_roundtrip():
    run_id = create_run_id("test")
    config = ExperimentConfig(run_name="phase2-smoke")

    config_path = save_experiment_config(config, run_id)
    assert config_path.exists()

    metadata = RunMetadata(
        run_id=run_id,
        run_name=config.run_name,
        phase_mode=config.phase_mode,
        seed=config.seed,
        started_at=1,
    )
    metadata_path = save_run_metadata(metadata)
    assert metadata_path.exists()


def test_mark_run_finished_sets_final_fields():
    run_id = create_run_id("test")
    metadata = RunMetadata(
        run_id=run_id,
        run_name="phase2-smoke",
        phase_mode="phase2",
        seed=42,
        started_at=1,
    )
    save_run_metadata(metadata)

    out_path = mark_run_finished(
        run_id,
        status="completed",
        metrics={"recall_at_10": 0.35},
        artifacts={"checkpoint": "data/processed/checkpoints/latest.pt"},
    )

    assert out_path.exists()
    payload = out_path.read_text(encoding="utf-8")
    assert "completed" in payload
    assert "recall_at_10" in payload


def test_runs_dir_is_used_for_artifacts():
    run_id = create_run_id("test")
    config = ExperimentConfig(run_name="phase2-smoke")
    config_path = save_experiment_config(config, run_id)
    assert str(config_path).startswith(str(RUNS_DIR))
