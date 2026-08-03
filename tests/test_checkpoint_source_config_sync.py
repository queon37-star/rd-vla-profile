import hashlib
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

from experiments.robot import openvla_utils
from experiments.robot.libero.run_libero_eval import GenerateConfig


def _snapshot(directory: Path):
    return {
        path.relative_to(directory).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("configured", [None, True])
def test_checkpoint_source_config_sync_defaults_to_enabled(
    monkeypatch,
    configured,
):
    calls = []
    cfg_values = {"pretrained_checkpoint": "/tmp/checkpoint"}
    if configured is not None:
        cfg_values["sync_checkpoint_source_config"] = configured
    cfg = SimpleNamespace(**cfg_values)

    monkeypatch.setattr(
        openvla_utils,
        "update_auto_map",
        lambda checkpoint: calls.append(("auto_map", checkpoint)),
    )
    monkeypatch.setattr(
        openvla_utils,
        "check_model_logic_mismatch",
        lambda checkpoint: calls.append(("model_logic", checkpoint)),
    )

    assert openvla_utils.prepare_checkpoint_source_config(cfg) is True
    assert calls == [
        ("auto_map", "/tmp/checkpoint"),
        ("model_logic", "/tmp/checkpoint"),
    ]


def test_checkpoint_source_config_sync_can_be_disabled(monkeypatch):
    cfg = SimpleNamespace(
        pretrained_checkpoint="/tmp/checkpoint",
        sync_checkpoint_source_config=False,
    )
    calls = []
    monkeypatch.setattr(
        openvla_utils,
        "update_auto_map",
        lambda checkpoint: calls.append(("auto_map", checkpoint)),
    )
    monkeypatch.setattr(
        openvla_utils,
        "check_model_logic_mismatch",
        lambda checkpoint: calls.append(("model_logic", checkpoint)),
    )

    assert openvla_utils.prepare_checkpoint_source_config(cfg) is False
    assert calls == []


def test_disabled_sync_preserves_checkpoint_snapshot(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type": "openvla"}\n',
        encoding="utf-8",
    )
    (checkpoint / "modeling_prismatic.py").write_text(
        "MODEL = 'checkpoint'\n",
        encoding="utf-8",
    )
    (checkpoint / "configuration_prismatic.py").write_text(
        "CONFIG = 'checkpoint'\n",
        encoding="utf-8",
    )
    before = _snapshot(checkpoint)

    cfg = SimpleNamespace(
        pretrained_checkpoint=str(checkpoint),
        sync_checkpoint_source_config=False,
    )
    assert openvla_utils.prepare_checkpoint_source_config(cfg) is False

    assert _snapshot(checkpoint) == before
    assert not list(checkpoint.glob("config.json.back.*"))
    assert not list(checkpoint.glob("modeling_*.py.back.*"))
    assert not list(checkpoint.glob("configuration_*.py.back.*"))


def test_get_vla_passes_disabled_sync_setting(monkeypatch):
    prepared = []

    class FakeVisionBackbone:
        def set_num_images_in_input(self, count):
            self.count = count

    class FakeVLA:
        def __init__(self):
            self.vision_backbone = FakeVisionBackbone()

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(
        openvla_utils,
        "model_is_on_hf_hub",
        lambda checkpoint: False,
    )
    monkeypatch.setattr(
        openvla_utils,
        "prepare_checkpoint_source_config",
        lambda cfg: prepared.append(cfg.sync_checkpoint_source_config),
    )
    for registry in (
        openvla_utils.AutoConfig,
        openvla_utils.AutoImageProcessor,
        openvla_utils.AutoProcessor,
        openvla_utils.AutoModelForVision2Seq,
    ):
        monkeypatch.setattr(registry, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        openvla_utils.AutoModelForVision2Seq,
        "from_pretrained",
        lambda *args, **kwargs: FakeVLA(),
    )
    monkeypatch.setattr(
        openvla_utils,
        "_load_dataset_stats",
        lambda model, checkpoint: None,
    )
    cfg = SimpleNamespace(
        pretrained_checkpoint="/tmp/checkpoint",
        sync_checkpoint_source_config=False,
        load_in_8bit=False,
        load_in_4bit=False,
        use_film=False,
        num_images_in_input=2,
    )

    model = openvla_utils.get_vla(cfg)

    assert prepared == [False]
    assert model.vision_backbone.count == 2


def test_generate_config_and_draccus_boolean_contract():
    assert GenerateConfig().sync_checkpoint_source_config is True

    parser = draccus.argparsing.ArgumentParser(
        config_class=GenerateConfig,
    )
    cfg = parser.parse_args(
        ["--sync_checkpoint_source_config", "False"]
    )

    assert cfg.sync_checkpoint_source_config is False
