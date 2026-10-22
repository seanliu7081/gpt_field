"""Evaluator-only identities for the three authorized LIBERO-10 episodes.

The policy receives only each official task sentence, never reset hashes or goals.
Hashes are the existing episode0/seed7/settle10/native512 reset references.
"""
from gpt_field.distance_tasks import resolve_distance_task

LIBERO10_TASK_KEYS = ('libero_10_t2', 'libero_10_t6', 'libero_10_t8')
RESET_SHA256 = {('libero_10_t2', 0): '7b854d9bd19a0d91a843980142f15b6586416ff51790e447b044d6c89d1eceef', ('libero_10_t6', 0): '59d7c04f412daf02dc8818807d8343bda70e6f4dcffa2dfdb16cf9cb35bd048e', ('libero_10_t8', 0): 'd07133dc3b5ced57c573b9148abfc1416f66e38b91618f8b0e83b09962cbc4bb'}


def expected_reset(task_key, episode_index, old_references):
    if task_key in LIBERO10_TASK_KEYS:
        return RESET_SHA256[(task_key, episode_index)]
    return old_references.get((task_key, episode_index))
