"""Display the latest process-local MPS samples, optionally refreshing live."""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results/reproduction/memory"


def latest_sample(path):
    # Read just the tail; a writer may still be appending the last JSON line.
    with path.open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 8192))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', default='core_nodeclass_mps_memory')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            print('\nMPS memory (MiB): tensor / driver / sampled peak tensor / sampled peak driver', flush=True)
            for path in sorted((ROOT / args.setting).glob('*.jsonl')):
                row = latest_sample(path)
                if row is None:
                    continue
                summary = path.with_suffix('.summary.json')
                age = time.time() - row['time_unix']
                state = json.loads(summary.read_text())['status'] if summary.exists() else f'last sample {age:.1f}s ago'
                values = ' / '.join(f'{row[key] / 2**20:.1f}' for key in (
                    'tensor_bytes', 'driver_bytes', 'sampled_peak_tensor_bytes', 'sampled_peak_driver_bytes'))
                print(f'{path.stem}: {values} [{state}]', flush=True)
            if not args.watch:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
