"""Artifact inspection tests; no network or author code is needed."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probe import weights


class Response(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def geturl(self):
        return "https://example.org/final.pt"


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.asset = {"url": "https://example.org/model.pt", "sha256": hashlib.sha256(b"abc").hexdigest()}

    def test_verified_cache_and_corruption(self):
        with patch.object(weights.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(b"abc", 3)) as fetch:
            path, info = weights.download(self.asset, self.root, 10, 1)
            self.assertEqual(info["sha256"], self.asset["sha256"])
            weights.download(self.asset, self.root, 10, 1)
            self.assertEqual(fetch.call_count, 1)
            path.write_bytes(b"bad")
            weights.download(self.asset, self.root, 10, 1)
            self.assertEqual(fetch.call_count, 2)

    def test_mutable_urls_are_refetched(self):
        self.asset.pop("sha256")
        with patch.object(weights.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(b"abc")) as fetch:
            for _ in range(2): weights.download(self.asset, self.root, 10, 1)
            self.assertEqual(fetch.call_count, 2)

    def test_pinned_hub_cache(self):
        self.asset = {"url": "https://huggingface.co/org/model/resolve/" + "a" * 40 + "/model.pt"}
        with patch.object(weights.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(b"abc")) as fetch:
            for _ in range(2): weights.download(self.asset, self.root, 10, 1)
            self.assertEqual(fetch.call_count, 1)

    def test_bad_downloads_leave_no_checkpoint(self):
        for data, length, limit in [(b"bad", None, 10), (b"abc", 20, 10), (b"abc", None, 2), (b"abc", 4, 10)]:
            with self.subTest(data=data, length=length, limit=limit):
                with patch.object(weights.urllib.request, "urlopen", return_value=Response(data, length)):
                    with self.assertRaises(ValueError):
                        weights.download(self.asset, self.root, limit, 1)
                self.assertFalse(list(self.root.rglob("*.partial")))
                self.assertFalse(list(self.root.rglob("checkpoint")))

    def test_declared_size_limit_prevents_network(self):
        with patch.object(weights.urllib.request, "urlopen") as fetch:
            with self.assertRaises(ValueError):
                weights.download(dict(self.asset, size_bytes=100), self.root, 10, 1)
            fetch.assert_not_called()


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        card = self.root / "content/models/example/index.md"
        card.parent.mkdir(parents=True)
        card.write_text('+++\n[[assets]]\nkey="weights"\nurl="https://example.org/model.pt"\n'
                        '[[assets]]\nkey="onnx"\nurl="https://example.org/model.onnx"\n+++\n')

    def test_selection_and_listing_never_download(self):
        self.assertEqual(len(weights.select_assets(self.root, ["example"], "weights")), 1)
        with self.assertRaises(ValueError): weights.select_assets(self.root, ["missing"])
        with self.assertRaises(ValueError): weights.select_assets(self.root, ["example"], "missing")
        report = self.root / "list.json"
        with patch.object(weights, "download") as download, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(weights.main(["--repo", str(self.root), "--all", "--list", "--output", str(report)]), 0)
            download.assert_not_called()
        self.assertEqual([r['status'] for r in json.loads(report.read_text())['results']], ['eligible', 'unsupported'])

    def test_requires_selection(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            weights.main([])

    def test_unsupported_links(self):
        for url in ['https://huggingface.co/o/m', 'https://github.com/o/m/blob/main/model.pt',
                    'https://example.org/model.zip', 'file:///tmp/a.pt',
                    'https://example.org/pytorch_model-00001-of-00002.bin']:
            self.assertIsNotNone(weights.unsupported_reason({'url': url}))

    def test_failure_is_reported_and_other_assets_continue(self):
        output = self.root / 'result.json'
        with patch.object(weights, 'download', side_effect=ValueError('checksum failure')), contextlib.redirect_stdout(io.StringIO()):
            code = weights.main(['--repo', str(self.root), '--all', '--run-root', str(self.root/'runs'), '--output', str(output)])
        self.assertEqual(code, 1)
        results = json.loads(output.read_text())['results']
        self.assertEqual([r['status'] for r in results], ['failed','unsupported'])
        self.assertIsNone(results[0]['runtime_parameter_count'])
        self.assertEqual(results[0]['candidates'], [])

    def test_worker_success_and_timeout_reports(self):
        output = self.root / 'result.json'
        arguments = ['--repo', str(self.root), '--model', 'example', '--asset', 'weights',
                     '--run-root', str(self.root/'runs'), '--output', str(output)]

        def worker(command, cwd, env, log, timeout):
            Path(command[-1]).write_text(json.dumps({
                'status': 'ok', 'torch_version': 'test', 'runtime_parameter_count': None,
                'candidates': [{'container': 'root', 'tensor_elements': 6}],
            }))
            return 0

        with patch.object(weights, 'download', return_value=(self.root/'checkpoint', {'sha256': 'a'*64})), \
                patch.object(weights, 'execute', side_effect=worker), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(weights.main(arguments), 0)
        result = json.loads(output.read_text())['results'][0]
        self.assertEqual(result['candidates'][0]['tensor_elements'], 6)
        self.assertEqual(result['sha256'], 'a'*64)
        with patch.object(weights, 'download', return_value=(self.root/'checkpoint', {})), \
                patch.object(weights, 'execute', side_effect=TimeoutError('worker timeout')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(weights.main(arguments), 1)
        self.assertIn('worker timeout', json.loads(output.read_text())['results'][0]['reason'])


try:
    import torch
except ImportError:
    torch = None


class Untrusted:
    def __reduce__(self):
        return os.system, ("echo unsafe",)


@unittest.skipIf(torch is None, "Run in environments/weights to test the restricted loader")
class WorkerTests(unittest.TestCase):
    def setUp(self):
        from probe.weight_worker import inspect
        self.inspect = inspect
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'weights.pt'

    def test_counts_buffers_scalar_empty_and_shared_names(self):
        shared = torch.zeros(2, 3)
        state = {'weight': shared, 'alias': shared, 'scalar': torch.tensor(2.),
                 'empty': torch.empty(0, 4), 'num_batches_tracked': torch.tensor(3)}
        torch.save(state, self.path)
        result = self.inspect(self.path)
        self.assertEqual(result['status'], 'ok')
        candidate = result['candidates'][0]
        self.assertEqual(candidate['tensor_elements'], 14)
        self.assertEqual(candidate['tensor_count'], 5)
        self.assertEqual(candidate['by_dtype']['torch.int64']['elements'], 1)
        self.assertIsNone(result['runtime_parameter_count'])

    def test_separate_candidates_exclude_optimizer(self):
        torch.save({'model': {'w': torch.zeros(2)}, 'ema': {'w': torch.zeros(3)},
                    'optimizer': {'w': torch.zeros(100)}}, self.path)
        result = self.inspect(self.path)
        self.assertEqual([(c['container'],c['tensor_elements']) for c in result['candidates']], [('model',2),('ema',3)])
        self.assertEqual(result['status'], 'ok')
        for name in ['state_dict', 'model_state_dict']:
            torch.save({name: {'w': torch.ones(4)}, 'epoch': 12}, self.path)
            self.assertEqual(self.inspect(self.path)['candidates'][0]['tensor_elements'], 4)

    def test_partial_and_empty_layouts_are_unresolved(self):
        for obj in [{}, {'optimizer': {'w': torch.ones(1)}},
                    {'model': {'w': torch.ones(1)}, 'ema': None},
                    {'w': torch.ones(1), 'extra': 'metadata'}]:
            torch.save(obj, self.path)
            self.assertEqual(self.inspect(self.path)['status'], 'unresolved')

    def test_rejects_custom_objects_without_execution(self):
        torch.save(Untrusted(), self.path)
        with patch('os.system') as execute:
            self.assertEqual(self.inspect(self.path)['status'], 'unresolved')
            execute.assert_not_called()

    def test_malformed_and_torchscript(self):
        self.path.write_bytes(b'not a checkpoint')
        self.assertEqual(self.inspect(self.path)['status'], 'unresolved')
        module = torch.jit.trace(torch.nn.Linear(2, 1), torch.ones(1, 2))
        torch.jit.save(module, str(self.path))
        with patch.object(torch, 'load') as load:
            self.assertEqual(self.inspect(self.path)['status'], 'unsupported')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
