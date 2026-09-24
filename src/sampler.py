"""Entity-aware batch sampler: no two records of the same S1 entity share a batch."""

from __future__ import annotations

import random
from collections import defaultdict


class EntityAwareBatchSampler:
    """Yield batches of example indices with at most one row per entity_id."""

    def __init__(self, entity_ids: list[str], batch_size: int, seed: int = 42):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.groups: dict[str, list[int]] = defaultdict(list)
        for i, eid in enumerate(entity_ids):
            self.groups[str(eid)].append(i)
        self.n_batches = max(len(entity_ids) // self.batch_size, 1)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        entities = list(self.groups.keys())
        rng.shuffle(entities)
        # one pointer per entity
        pointers = {e: 0 for e in entities}
        shuffled_items = {e: self.groups[e][:] for e in entities}
        for e in entities:
            rng.shuffle(shuffled_items[e])

        ei = 0
        n_e = len(entities)
        produced = 0
        while produced < self.n_batches:
            batch = []
            seen = 0
            while len(batch) < self.batch_size and seen < n_e:
                e = entities[ei % n_e]
                ei += 1
                seen += 1
                items = shuffled_items[e]
                p = pointers[e]
                if p >= len(items):
                    continue
                batch.append(items[p])
                pointers[e] = p + 1
            if len(batch) < 2:
                break
            produced += 1
            yield batch
            if ei > n_e * 4 and produced < 2:
                break
            # reshuffle exhausted entities
            if ei % max(n_e, 1) == 0:
                rng.shuffle(entities)
                for e in entities:
                    if pointers[e] >= len(shuffled_items[e]):
                        rng.shuffle(shuffled_items[e])
                        pointers[e] = 0
