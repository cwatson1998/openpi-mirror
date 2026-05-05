# LIBERO Spatial Four Bowls

`libero_spatial_four_bowls` is a local LIBERO eval suite derived from
`libero_spatial`. It keeps the same 10 task prompts and goal predicates, but
each task starts with four black bowls instead of two. The prompted / goal bowl
is still `akita_black_bowl_1`; the other bowls are distractors placed at other
common spatial relations.

The benchmark assets live in the LIBERO submodule:

- `third_party/libero/libero/libero/bddl_files/libero_spatial_four_bowls`
- `third_party/libero/libero/libero/init_files/libero_spatial_four_bowls`

Each task has 50 generated initial states.

## Scripted V2 Policy

`examples/libero/scripted_spatial_four_bowls_policy.py` implements the
four-bowls v2 scripted policy. It uses low-dimensional simulator state only:
robot end-effector proprioception plus live object positions from
`env.env.object_states_dict`.

The policy is routed only when running:

```bash
--args.task-suite-name libero_spatial_four_bowls --args.policy-version v2
```

The original `libero_spatial` v2 implementation in
`examples/libero/scripted_spatial_policy.py` is unchanged.

The four-bowls policy:

- reads all four bowl positions plus the plate, cookie box, ramekin, cabinet,
  and stove positions
- uses the task language to choose relation-specific grasp/place behavior
- tracks the observed bowl-to-gripper offset after grasp for more accurate
  placement
- retries failed placements when the first attempt does not satisfy the LIBERO
  success predicate

## Running Eval

Use the LIBERO Python 3.10 environment:

```bash
PYTHONPATH=third_party/libero MUJOCO_GL=osmesa examples/libero/.venv/bin/python \
  examples/libero/eval_scripted_spatial.py \
  --args.task-suite-name libero_spatial_four_bowls \
  --args.policy-version v2 \
  --args.num-trials-per-task 5
```

The scripted eval runner automatically uses a longer default horizon for
`libero_spatial_four_bowls` because retry rollouts can be longer than standard
`libero_spatial` rollouts.

## Current Baseline

The current four-bowls v2 scripted policy reached:

```text
48 / 50 successes = 96%
```

on 5 trials per task. The only failures in that run were two initial states from
the top-drawer task. Results were written to:

```text
data/libero/runs/scripted_four_bowls_v2_5ep_final/results.json
```
