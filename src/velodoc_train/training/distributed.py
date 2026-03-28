import os


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0
