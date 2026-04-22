# Git Remote Settings

## Main repo

Path: `/home/christopher/Documents/openpi-finetune/openpi`

### `git remote -v`

```text
origin       git@github.com:cwatson1998/openpi-mirror.git (fetch)
origin       git@github.com:cwatson1998/openpi-mirror.git (push)
public-fork  git@github.com:cwatson1998/openpi.git (fetch)
public-fork  git@github.com:cwatson1998/openpi.git (push)
```

### `git config --local --get-regexp '^remote\.'`

```text
remote.public-fork.url git@github.com:cwatson1998/openpi.git
remote.public-fork.fetch +refs/heads/*:refs/remotes/public-fork/*
remote.origin.url git@github.com:cwatson1998/openpi-mirror.git
remote.origin.fetch +refs/heads/*:refs/remotes/origin/*
```

## `third_party/libero`

Path: `/home/christopher/Documents/openpi-finetune/openpi/third_party/libero`

### `git remote -v`

```text
mirror  git@github.com:cwatson1998/LIBERO-mirror.git (fetch)
mirror  git@github.com:cwatson1998/LIBERO-mirror.git (push)
origin  https://github.com/Lifelong-Robot-Learning/LIBERO.git (fetch)
origin  https://github.com/Lifelong-Robot-Learning/LIBERO.git (push)
```

### `git config --local --get-regexp '^remote\.'`

```text
remote.origin.url https://github.com/Lifelong-Robot-Learning/LIBERO.git
remote.origin.fetch +refs/heads/*:refs/remotes/origin/*
remote.mirror.url git@github.com:cwatson1998/LIBERO-mirror.git
remote.mirror.fetch +refs/heads/*:refs/remotes/mirror/*
```
