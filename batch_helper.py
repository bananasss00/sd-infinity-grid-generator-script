from collections import namedtuple, OrderedDict
from copy import copy
import re, random, sys, os
from itertools import groupby
from typing import Any
from modules.processing import StableDiffusionProcessing

script_path = os.path.dirname(os.path.abspath(__file__))
module_directory = os.path.join(script_path, 'scripts')

def apply_model(p, v):
    if module_directory not in sys.path:
        sys.path.append(module_directory)
    from infinity_grid import apply_model as cn
    return cn(p, v)

def apply_restore_faces(p, v):
    if module_directory not in sys.path:
        sys.path.append(module_directory)
    from infinity_grid import apply_restore_faces as cn
    return cn(p, v)


class BatchHelper:
    BATCH_BY_PARAMS = ['prompt', 'negative_prompt', 'seed', 'subseed']
    BATCH_BY_PARAMS_SRC = ['prompt', 'negative prompt', 'seed', 'subseed']

    def __init__(self) -> None:
        self.cleanup()

    def _split_into_batches(self, lst, batch_size):
        return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

    def _get_key(self, item):
        p = item[0]
        params = copy(item[1])
        params['seed'] = params['subseed'] = 0
        params['prompt'] = tuple(re.findall(r"<([^>]+)>", p.prompt))
        params['negative prompt'] = tuple(re.findall(r"<([^>]+)>", p.negative_prompt))

        def sort_by_models(x):
            keys_order = ['model', 'vae', 'prompt', 'negative prompt']
            return keys_order.index(x[0]) if x[0] in keys_order else len(keys_order) + 1
        
        params = OrderedDict(sorted(params.items(), key=sort_by_models))
        return '_'.join(f'{k}_{v}' for k, v in params.items())

    
    def cleanup(self):
        self._laterun: dict[StableDiffusionProcessing, tuple[str, Any]] = {}

    def apply_to_hook(self, p, name, val) -> bool:
        if name in BatchHelper.BATCH_BY_PARAMS_SRC:
            return False

        if not p in self._laterun:
            self._laterun[p] = []

        self._laterun[p].append((name, val))
        return True
    
    def laterun(self):
        return self._laterun

    def group_batches(self, prompts, sets, max_batch_size: int | None = 16, debug_batch = False):
        data = [(p, sets[p].params) for p in prompts]
        if max_batch_size == None: # if not defined use from ui batch size
            max_batch_size = max(p.batch_size for p in prompts)

        # random.shuffle(data) # debug
        sorted_data = sorted(data, key=self._get_key)
        grouped_data = [list(group) for _, group in groupby(sorted_data, key=self._get_key)]

        groups_final = []
        for group in grouped_data:
            if len(group) > max_batch_size:
                for groups in self._split_into_batches(group, max_batch_size):
                    groups_final.append([g[0] for g in groups])
            else:
                groups_final.append([g[0] for g in group])

        if debug_batch:
            for group in groups_final:
                print('--- GROUP ---')
                for p in group:
                    print(p.seed, p.prompt, sets[p].params)
            input('Press Enter for continue...')
        
        merged_prompts = []
        merged_sets = {}
        for pg in groups_final:
            merged_prompt = copy(pg[0])
            merged_prompt.prompt = [p.prompt for p in pg] if len(pg) > 1 else pg[0].prompt
            merged_prompt.negative_prompt = [p.negative_prompt for p in pg] if len(pg) > 1 else pg[0].negative_prompt
            merged_prompt.seed = [p.seed for p in pg] if len(pg) > 1 else pg[0].seed
            merged_prompt.subseed = [p.subseed for p in pg] if len(pg) > 1 else pg[0].subseed
            merged_prompt.batch_size = len(pg)
            merged_prompts.append(merged_prompt)

            merged_sets[merged_prompt] = []
            for p in pg:
                merged_sets[merged_prompt].append(sets[p])

            if pg[0] in self._laterun:
                self._laterun[merged_prompt] = self._laterun[pg[0]]

        return merged_prompts, merged_sets