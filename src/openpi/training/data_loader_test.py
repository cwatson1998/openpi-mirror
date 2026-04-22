import dataclasses
import json

import jax
import pytest

from openpi.models import pi0
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_select_task_prompts_uses_filtered_libero_subset(tmp_path):
    path = tmp_path / "logic.json"
    path.write_text(
        json.dumps(
            {
                "dataset_name": "physical-intelligence/libero",
                "suite_name": "libero_10",
                "tasks": [
                    {
                        "task_id": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
                        "task_instruction": "turn on the stove and put the moka pot on it",
                        "bddl_file": "/tmp/example.bddl",
                        "goal_description": "(And (Turnon flat_stove) (On moka_pot flat_stove))",
                        "logic_task_description": "(And (Turnon flat_stove) (On moka_pot flat_stove))",
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )

    prompt_map = _data_loader._select_task_prompts(  # noqa: SLF001
        {
            0: {"task": "turn on the stove and put the moka pot on it"},
            1: {"task": "pick up the ketchup and place it in the basket"},
        },
        task_filters=("turn on the stove and put the moka pot on it",),
        task_description_field=None,
        task_description_path=str(path),
        prompt_from_task=True,
    )

    assert prompt_map == {0: "(And (Turnon flat_stove) (On moka_pot flat_stove))"}


def test_select_task_prompts_uses_dataset_task_description_field():
    prompt_map = _data_loader._select_task_prompts(  # noqa: SLF001
        {
            0: {
                "task": "turn on the stove and put the moka pot on it",
                "task_description": "natural language but stored in metadata",
                "task_description_1": "(And (Turnon flat_stove) (On moka_pot flat_stove))",
            },
            1: {
                "task": "pick up the ketchup and place it in the basket",
                "task_description_1": "(And (In ketchup_1 basket_1_contain_region))",
            },
        },
        task_filters=("turn on the stove and put the moka pot on it",),
        task_description_field="task_description_1",
        task_description_path=None,
        prompt_from_task=True,
    )

    assert prompt_map == {0: "(And (Turnon flat_stove) (On moka_pot flat_stove))"}


def test_select_task_prompts_raises_when_dataset_task_description_field_is_missing():
    with pytest.raises(ValueError, match="Task description field `task_description_1`"):
        _data_loader._select_task_prompts(  # noqa: SLF001
            {
                0: {"task": "turn on the stove and put the moka pot on it"},
            },
            task_filters=(),
            task_description_field="task_description_1",
            task_description_path=None,
            prompt_from_task=True,
        )
