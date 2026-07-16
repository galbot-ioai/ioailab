# Tutorial

Welcome to ioailab, the IOAI-oriented simulation platform built by the Galbot
simulation team. This tutorial shows how to build a task step by step, starting
from a simple PickCube task and ending with larger compound tasks that run
through several phases.

The tutorial is split into three chapters. Read them in order: the first chapter
teaches the basic task-generation loop, and the later chapters reuse that loop
for component tasks and compound tasks.

```text
simple task
-> component tasks
-> compound task
```

- [Chapter 1: Build and Activate a Simple Task](tutorials/01_build_and_activate_simple_task.md)
- [Chapter 2: Build PickToShelf From Component Tasks](tutorials/02_pick_to_shelf_components_to_compound.md)
- [Chapter 3: Build SortToShelf With Task Options and Sequence Agents](tutorials/03_sort_to_shelf_options_and_sequence_agent.md)

The main workflow is iterative. A new task should become runnable before it
becomes complete: first make IsaacLab construct the environment, then run one
episode, refine the MDP and reset behavior, collect data, and only then add
Mimic, training, evaluation, or compound-task structure. It is normal to return
to an earlier step when a later step exposes a weak termination, observation, or
initial-state definition.

Keep task MDP functions under semantic owners such as `events.py`,
`observations.py`, `rewards.py`, and `terminations.py`.
Do not add a generic `terms.py` bucket.

```text
minimal task
-> single-env smoke
-> optional teleop
-> refine MDP/reset
-> multi-env smoke
-> collect data
-> optional mimic dataset expansion
-> train
-> evaluate
-> revisit task definition
```
