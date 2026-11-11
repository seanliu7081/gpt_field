"""Render exact historical relayout resets; no learned-policy action or inference."""
import json
from pathlib import Path
from dataclasses import replace
import sys
import hashlib
import numpy as np
from PIL import Image
ROOT = Path('/workspace/gpt_field')
sys.path.insert(0, str(ROOT / 'src'))
from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config
from gpt_field.libero import _LiberoWorker
from gpt_field.agent_speedup_v2_user_check import initial_observation_sha256
from gpt_field.observation import PolicyObservation
from gpt_field.contact_skills import StepView
BASE = Path(__file__).resolve().parent
manifest = json.loads((BASE / 'recovery_manifest.json').read_text())
results = []
workers = []
for task in (0, 1, 6):
    folder = BASE / f'task_{task:02d}'
    cfg, _ = build_v2_sim_config(f'libero_10_t{task}')
    cfg = replace(cfg, initial_state_override=str(folder / 'initial_state.npy'), collision_check_mode='none')
    worker = _LiberoWorker(cfg)
    workers.append(worker)
    try:
        response = worker.reset(task, 0)
        obs = response['observation']
        actual = initial_observation_sha256(obs)
        expected = manifest['tasks'][str(task)]['expected_initial_observation_sha256']
        (folder / 'validated_initial_observation.json').write_text(json.dumps(obs, sort_keys=True) + '\n')
        state = np.asarray(worker.env.sim.get_state().flatten()).copy()
        np.save(folder / 'expected_settled_state.npy', state, allow_pickle=False)
        view = StepView(PolicyObservation.from_dict(obs))
        pixels_match = {}
        for name, pixels in [('external', view.external_rgb_upright), ('wrist', view.wrist_rgb_upright)]:
            Image.fromarray(pixels).save(folder / f'validated_{name}_512.png')
            if name == 'external':
                saved = ROOT / 'output/visualizations/libero10-relayout-comparison-20261105' / f'task_{task:02d}' / 'relayout_external.png'
                pixels_match[name] = bool(np.array_equal(pixels, np.array(Image.open(saved).convert('RGB'))))
        controller = worker.env.robots[0].controller
        result = {'task_id': task, 'observation_sha256': actual, 'expected_observation_sha256': expected,
                  'observation_hash_match': actual == expected, 'saved_images_pixel_match': pixels_match,
                  'raw_libero_success': bool(worker.env.check_success()), 'settle_steps': cfg.settle_steps,
                  'settle_action': [0,0,0,0,0,0,-1], 'simulator_seed': cfg.seed,
                  'expected_settled_state_sha256': hashlib.sha256((folder/'expected_settled_state.npy').read_bytes()).hexdigest(),
                  'controller_output_min': controller.output_min.tolist(), 'controller_output_max': controller.output_max.tolist(),
                  'bddl': manifest['tasks'][str(task)]['official_bddl_source'],
                  'passed': actual == expected and all(pixels_match.values()) and not worker.env.check_success()}
        (folder / 'reset_validation.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        print(json.dumps(result), flush=True)
    finally:
        if worker.env is not None:
            worker.env.close()
            worker.env = None
for worker in workers:
    worker.close()
summary = {'passed': all(row['passed'] for row in results), 'tasks': results,
           'policy_actions': 0, 'model_calls': 0, 'preparation_settling_actions': 30}
(BASE / 'reset_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
assert summary['passed'], 'Historical reset mismatch; inspect reports before evaluation'
