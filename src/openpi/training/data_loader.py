from collections.abc import Iterator, Sequence
import json
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.libero as libero_utils
import openpi.training.libero_logic as libero_logic
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def _normalize_task_name(task_name: str) -> str:
    return " ".join(task_name.strip().replace("_", " ").split()).casefold()


def _load_lerobot_task_metadata(dataset_meta: lerobot_dataset.LeRobotDatasetMetadata) -> dict[int, dict[str, str]]:
    """Load the full LeRobot task records, preserving any alternate prompt fields.

    LeRobot exposes `dataset_meta.tasks` as a simple `task_index -> task` mapping, which is
    enough for the default prompt path. For custom datasets we also want to support
    dataset-native prompt variants (for example `task_description_1`) stored alongside the
    canonical `task` string in `meta/tasks.jsonl`. Existing datasets remain compatible
    because we synthesize a minimal record when only the canonical field is present.
    """

    tasks_path = dataset_meta.root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return {int(task_index): {"task": task} for task_index, task in dataset_meta.tasks.items()}

    task_metadata: dict[int, dict[str, str]] = {}
    for line_number, line in enumerate(tasks_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        record = json.loads(line)
        if "task_index" not in record:
            raise ValueError(f"Missing `task_index` in {tasks_path}:{line_number}")

        task_index = int(record["task_index"])
        task = record.get("task")
        if not isinstance(task, str):
            raise ValueError(f"Missing string `task` in {tasks_path}:{line_number}")

        string_fields = {key: value for key, value in record.items() if isinstance(value, str)}
        string_fields["task"] = task
        task_metadata[task_index] = string_fields

    # Preserve backwards compatibility with older datasets even if their `tasks.jsonl`
    # omits entries that LeRobot already surfaced through `dataset_meta.tasks`.
    for task_index, task in dataset_meta.tasks.items():
        task_metadata.setdefault(int(task_index), {"task": task})

    return task_metadata


def _select_task_prompts(
    dataset_task_metadata: dict[int, dict[str, str]],
    *,
    task_filters: Sequence[str],
    task_description_field: str | None,
    task_description_path: str | None,
    prompt_from_task: bool,
) -> dict[int, str] | None:
    dataset_tasks = {task_index: fields["task"] for task_index, fields in dataset_task_metadata.items()}
    selected_tasks = dataset_tasks
    if task_filters:
        allowed_tasks = {_normalize_task_name(task) for task in task_filters}
        selected_tasks = {
            int(task_index): task_instruction
            for task_index, task_instruction in dataset_tasks.items()
            if _normalize_task_name(task_instruction) in allowed_tasks
        }

    if task_description_path is not None:
        return libero_logic.build_task_prompt_map_from_dataset_tasks(selected_tasks, task_description_path)

    if task_description_field is not None:
        missing = [
            dataset_tasks[task_index]
            for task_index in selected_tasks
            if task_description_field not in dataset_task_metadata[task_index]
        ]
        if missing:
            raise ValueError(
                f"Task description field `{task_description_field}` was not found for all selected dataset tasks.\n"
                "Missing values for:\n"
                + "\n".join(f"  - {task}" for task in missing)
            )

        return {
            int(task_index): dataset_task_metadata[task_index][task_description_field] for task_index in selected_tasks
        }

    if prompt_from_task:
        return selected_tasks

    return None


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=data_config.local_files_only)
    dataset_task_metadata = _load_lerobot_task_metadata(dataset_meta)
    episode_indices = None
    if data_config.task_filters:
        available_task_names = {fields["task"].casefold(): fields["task"] for fields in dataset_task_metadata.values()}
        missing_tasks = sorted(task for task in data_config.task_filters if task.casefold() not in available_task_names)
        if missing_tasks:
            available = "\n".join(f"  - {task}" for task in available_task_names.values())
            raise ValueError(
                f"Task filters not found in dataset {repo_id}: {missing_tasks}\nAvailable tasks:\n{available}"
            )
        episode_indices = libero_utils.select_episode_indices(dataset_meta.episodes, data_config.task_filters)
        if not episode_indices:
            raise ValueError(f"No episodes matched the requested task filters for dataset {repo_id}.")

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        episodes=episode_indices,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        local_files_only=data_config.local_files_only,
    )

    task_prompts = _select_task_prompts(
        dataset_task_metadata,
        task_filters=data_config.task_filters,
        task_description_field=data_config.task_description_field,
        task_description_path=data_config.task_description_path,
        prompt_from_task=data_config.prompt_from_task,
    )

    if task_prompts is not None:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(task_prompts)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
