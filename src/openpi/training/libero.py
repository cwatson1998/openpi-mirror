from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import pathlib

LIBERO_RAW_DATASETS = {
    "libero_10": "libero_10_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_spatial": "libero_spatial_no_noops",
}

LIBERO_TASK_IDS = {
    "libero_spatial": (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ),
    "libero_spatial_four_bowls": (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ),
    "libero_object": (
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ),
    "libero_goal": (
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ),
    "libero_10": (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ),
}


@dataclasses.dataclass(frozen=True)
class LiberoTaskSplit:
    suite_name: str
    train_task_indices: tuple[int, ...]
    eval_task_indices: tuple[int, ...]
    train_task_instructions: tuple[str, ...]
    eval_task_instructions: tuple[str, ...]


def _task_id_to_instruction(task_id: str) -> str:
    if task_id.endswith(".bddl"):
        task_id = task_id[:-5]

    if task_id and task_id[0].isupper():
        scene_idx = task_id.find("SCENE")
        if scene_idx == -1:
            raise ValueError(f"Unexpected LIBERO task id: {task_id}")
        offset = 8 if "SCENE10" in task_id else 7
        return " ".join(task_id[scene_idx + offset :].split("_"))

    return " ".join(task_id.split("_"))


def _normalize_task_name(task_name: str) -> str:
    task_name = task_name.strip()
    if not task_name:
        raise ValueError("Task name cannot be empty.")
    if task_name.endswith(".bddl"):
        task_name = task_name[:-5]
    if "_" in task_name:
        task_name = _task_id_to_instruction(task_name)
    return " ".join(task_name.split()).casefold()


def get_libero_task_instructions(suite_name: str) -> tuple[str, ...]:
    if suite_name not in LIBERO_TASK_IDS:
        raise ValueError(f"Unsupported LIBERO suite: {suite_name}")
    return tuple(_task_id_to_instruction(task_id) for task_id in LIBERO_TASK_IDS[suite_name])


def get_libero_task_table(suite_name: str) -> list[tuple[int, str, str]]:
    task_ids = LIBERO_TASK_IDS[suite_name]
    task_instructions = get_libero_task_instructions(suite_name)
    return list(zip(range(len(task_ids)), task_ids, task_instructions, strict=True))


def resolve_task_instructions(
    suite_name: str,
    *,
    task_indices: Sequence[int] = (),
    task_names: Sequence[str] = (),
) -> tuple[str, ...]:
    task_table = get_libero_task_table(suite_name)
    by_normalized_name = {}
    for _, task_id, instruction in task_table:
        by_normalized_name[_normalize_task_name(task_id)] = instruction
        by_normalized_name[_normalize_task_name(instruction)] = instruction

    selected: list[str] = []
    for task_index in task_indices:
        if task_index < 0 or task_index >= len(task_table):
            raise ValueError(f"Task index {task_index} is out of range for {suite_name}.")
        selected.append(task_table[task_index][2])

    for task_name in task_names:
        normalized = _normalize_task_name(task_name)
        if normalized not in by_normalized_name:
            available = "\n".join(
                f"  {task_index}: {instruction}" for task_index, _, instruction in task_table
            )
            raise ValueError(f"Unknown LIBERO task '{task_name}' for suite {suite_name}.\nAvailable tasks:\n{available}")
        selected.append(by_normalized_name[normalized])

    if not selected:
        return tuple(task_table_item[2] for task_table_item in task_table)

    seen = set()
    deduped = []
    for task_name in selected:
        normalized = _normalize_task_name(task_name)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(task_name)
    return tuple(deduped)


def write_task_split(
    output_path: str | pathlib.Path,
    *,
    suite_name: str,
    train_task_indices: Sequence[int] = (),
    train_task_names: Sequence[str] = (),
    eval_task_indices: Sequence[int] = (),
    eval_task_names: Sequence[str] = (),
    use_remaining_tasks_for_eval: bool = True,
) -> LiberoTaskSplit:
    train_task_instructions = resolve_task_instructions(
        suite_name,
        task_indices=train_task_indices,
        task_names=train_task_names,
    )
    train_instruction_set = {_normalize_task_name(task) for task in train_task_instructions}
    all_task_instructions = get_libero_task_instructions(suite_name)

    if eval_task_indices or eval_task_names:
        eval_task_instructions = resolve_task_instructions(
            suite_name,
            task_indices=eval_task_indices,
            task_names=eval_task_names,
        )
    elif use_remaining_tasks_for_eval:
        eval_task_instructions = tuple(
            task for task in all_task_instructions if _normalize_task_name(task) not in train_instruction_set
        )
    else:
        eval_task_instructions = ()

    eval_instruction_set = {_normalize_task_name(task) for task in eval_task_instructions}
    overlap = train_instruction_set & eval_instruction_set
    if overlap:
        raise ValueError(f"Train/eval split overlap detected: {sorted(overlap)}")

    instruction_to_index = {
        _normalize_task_name(task_instruction): task_index
        for task_index, task_instruction in enumerate(all_task_instructions)
    }
    split = LiberoTaskSplit(
        suite_name=suite_name,
        train_task_indices=tuple(instruction_to_index[_normalize_task_name(task)] for task in train_task_instructions),
        eval_task_indices=tuple(instruction_to_index[_normalize_task_name(task)] for task in eval_task_instructions),
        train_task_instructions=tuple(train_task_instructions),
        eval_task_instructions=tuple(eval_task_instructions),
    )

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataclasses.asdict(split), indent=2) + "\n")
    return split


def load_task_split(path: str | pathlib.Path) -> LiberoTaskSplit:
    raw = json.loads(pathlib.Path(path).read_text())
    return LiberoTaskSplit(
        suite_name=raw["suite_name"],
        train_task_indices=tuple(raw.get("train_task_indices", ())),
        eval_task_indices=tuple(raw.get("eval_task_indices", ())),
        train_task_instructions=tuple(raw.get("train_task_instructions", ())),
        eval_task_instructions=tuple(raw.get("eval_task_instructions", ())),
    )


def resolve_task_filters(
    *,
    task_suite_name: str | None = None,
    task_indices: Sequence[int] = (),
    task_names: Sequence[str] = (),
    task_split_file: str | None = None,
    task_split: str = "train",
) -> tuple[str, ...]:
    if task_split not in {"train", "eval", "all"}:
        raise ValueError(f"Unsupported LIBERO task split: {task_split}")

    if task_split_file is not None:
        if task_suite_name is not None or task_indices or task_names:
            raise ValueError("Use either direct task selection or --task_split_file, not both.")
        split = load_task_split(task_split_file)
        if task_split == "train":
            return split.train_task_instructions
        if task_split == "eval":
            return split.eval_task_instructions
        return tuple(dict.fromkeys((*split.train_task_instructions, *split.eval_task_instructions)))

    if task_suite_name is None:
        return ()

    if not task_indices and not task_names and task_suite_name not in LIBERO_TASK_IDS:
        return ()

    return resolve_task_instructions(
        task_suite_name,
        task_indices=task_indices,
        task_names=task_names,
    )


def select_episode_indices(episodes: Sequence[dict], task_filters: Sequence[str]) -> list[int]:
    allowed_tasks = {_normalize_task_name(task) for task in task_filters}
    selected = []
    for episode in episodes:
        episode_tasks = {_normalize_task_name(task) for task in episode.get("tasks", ())}
        if episode_tasks & allowed_tasks:
            selected.append(int(episode["episode_index"]))
    return selected
