"""Compare HF/TP smoke caches on identical source records (CPU only)."""
import argparse
import json

import torch
from deepspec.data.target_cache_dataset import CacheDataset
from deepspec.data.ngram_cache import NgramCacheReader


def compare(reference, candidate, max_samples):
    datasets = [CacheDataset(p) for p in (reference, candidate)]
    try:
        if len(datasets[0]) != len(datasets[1]):
            raise ValueError('Different sample counts; use the same source range and filtering config')
        for key in ('target_layer_ids', 'hidden_size', 'chat_template_sha256', 'aux_reduction'):
            if datasets[0].manifest.get(key) != datasets[1].manifest.get(key):
                raise ValueError(f'Cache metadata differs: {key}')
        sides = [NgramCacheReader(p, d.manifest) if 'ngram_embedding' in d.manifest.get('extra_features', {})
                 else None for p, d in zip((reference, candidate), datasets)]
        if (sides[0] is None) != (sides[1] is None):
            raise ValueError('Only one cache contains ngram embeddings')
        metrics = {}
        count = min(len(datasets[0]), max_samples)
        if count == 0:
            raise ValueError('Empty cache')
        for i in range(count):
            rows = [d[i] for d in datasets]
            for key in ('input_ids', 'loss_mask'):
                if not torch.equal(rows[0][key], rows[1][key]):
                    raise ValueError(f'Sample {i}: {key} mismatch; features are not aligned')
            if sides[0] is not None:
                for row, side in zip(rows, sides):
                    row['ngram_embedding'] = side.read(i, seq_len=row['input_ids'].numel())
            for key in ('target_hidden_states', 'target_last_hidden_states', 'ngram_embedding'):
                if key not in rows[0]:
                    continue
                a, b = rows[0][key].float(), rows[1][key].float()
                if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError(f'Sample {i}: invalid {key}')
                # Chunk by tokens to avoid large float64 temporaries.
                m = metrics.setdefault(key, dict(elements=0, different=0, squared_error=0.,
                    reference_energy=0., candidate_energy=0., dot=0., max_abs=0.))
                for x, y in zip(a.split(128), b.split(128)):
                    x, y = x.double(), y.double()
                    diff = x-y
                    m['elements'] += x.numel()
                    m['different'] += int((x != y).sum())
                    m['squared_error'] += float(diff.square().sum())
                    m['reference_energy'] += float(x.square().sum())
                    m['candidate_energy'] += float(y.square().sum())
                    m['dot'] += float((x*y).sum())
                    m['max_abs'] = max(m['max_abs'], float(diff.abs().max()))
        result = {'samples_compared': count, 'features': {}}
        for key, m in metrics.items():
            result['features'][key] = dict(
                max_abs=m['max_abs'], rmse=(m['squared_error']/m['elements'])**.5,
                relative_l2=(m['squared_error']/max(m['reference_energy'], 1e-30))**.5,
                cosine=m['dot']/max((m['reference_energy']*m['candidate_energy'])**.5, 1e-30),
                different_fraction=m['different']/m['elements'])
        return result
    finally:
        for dataset in datasets:
            dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--max-samples', type=int, default=32)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error('max-samples must be positive')
    print(json.dumps(compare(args.reference, args.candidate, args.max_samples), indent=2))
