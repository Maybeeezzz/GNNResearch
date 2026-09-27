import json

import pytest

from paper_reproduction.experiments.mps_memory import MPSMemoryMonitor


class Backend:
    tensor = 100
    driver = 200

    def current_allocated_memory(self):
        return self.tensor

    def driver_allocated_memory(self):
        return self.driver

    def recommended_max_memory(self):
        return 1000


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf')])
def test_invalid_interval(tmp_path, interval):
    with pytest.raises(ValueError):
        MPSMemoryMonitor(tmp_path / 'memory.jsonl', interval, Backend())


def test_peaks_flush_and_error_cleanup(tmp_path):
    path = tmp_path / 'memory.jsonl'
    backend = Backend()
    with pytest.raises(RuntimeError, match='training failed'):
        with MPSMemoryMonitor(path, 60, backend) as monitor:
            assert json.loads(path.read_text())['tensor_bytes'] == 100
            backend.tensor, backend.driver = 300, 500
            monitor.sample()
            backend.tensor, backend.driver = 50, 150
            raise RuntimeError('training failed')
    summary = json.loads(path.with_suffix('.summary.json').read_text())
    assert summary['sampled_peak_tensor_bytes'] == 300
    assert summary['sampled_peak_driver_bytes'] == 500
    assert summary['status'] == 'failed'
    assert len(path.read_text().splitlines()) == 3
    assert not monitor.thread.is_alive()
    with pytest.raises(FileExistsError):
        with MPSMemoryMonitor(path, 60, backend):
            pass


def test_latest_sample_ignores_partial_write(tmp_path):
    from paper_reproduction.experiments.watch_mps_memory import latest_sample
    path = tmp_path / 'memory.jsonl'
    path.write_text('{"tensor_bytes": 100}\n{"tensor_bytes":')
    assert latest_sample(path) == {'tensor_bytes': 100}


def test_sampler_failure_is_not_silent(tmp_path):
    backend = Backend()
    path = tmp_path / 'memory.jsonl'
    with pytest.raises(RuntimeError, match='sampling failed'):
        with MPSMemoryMonitor(path, 60, backend) as monitor:
            monitor.error = OSError('counter unavailable')
    assert json.loads(path.with_suffix('.summary.json').read_text())['status'] == 'failed'
