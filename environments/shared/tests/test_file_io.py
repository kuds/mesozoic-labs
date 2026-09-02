"""Tests for atomic file-write helpers."""

import os
import stat

import numpy as np
import pytest

from environments.shared.file_io import atomic_copy, atomic_savez, atomic_write_text


class TestAtomicCopy:
    def test_copies_content(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"hello world")
        dst = tmp_path / "out" / "dst.bin"

        atomic_copy(src, dst)

        assert dst.read_bytes() == b"hello world"

    def test_overwrites_existing(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"new content")
        dst = tmp_path / "dst.bin"
        dst.write_bytes(b"old content")

        atomic_copy(src, dst)

        assert dst.read_bytes() == b"new content"

    def test_no_temp_files_left_behind(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.bin"

        atomic_copy(src, dst)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["dst.bin", "src.bin"]

    def test_missing_src_raises_and_preserves_dst(self, tmp_path):
        dst = tmp_path / "dst.bin"
        dst.write_bytes(b"intact")

        with pytest.raises(FileNotFoundError):
            atomic_copy(tmp_path / "nope.bin", dst)

        assert dst.read_bytes() == b"intact"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dst.bin"]


class TestAtomicSavez:
    def test_roundtrip(self, tmp_path):
        dst = tmp_path / "arrays.npz"
        timesteps = np.arange(5)
        results = np.random.rand(5, 3)

        atomic_savez(dst, timesteps=timesteps, results=results)

        data = np.load(str(dst))
        np.testing.assert_array_equal(data["timesteps"], timesteps)
        np.testing.assert_array_equal(data["results"], results)

    def test_overwrites_previous_archive(self, tmp_path):
        dst = tmp_path / "arrays.npz"
        atomic_savez(dst, a=np.array([1]))
        atomic_savez(dst, a=np.array([1, 2, 3]))

        data = np.load(str(dst))
        assert data["a"].shape == (3,)

    def test_no_temp_files_left_behind(self, tmp_path):
        dst = tmp_path / "arrays.npz"
        atomic_savez(dst, a=np.array([1]))

        assert [p.name for p in tmp_path.iterdir()] == ["arrays.npz"]


class TestAtomicWriteText:
    def test_writes_text_and_creates_parents(self, tmp_path):
        dst = tmp_path / "out" / "record.json"

        atomic_write_text(dst, '{"a": 1}\n')

        assert dst.read_text(encoding="utf-8") == '{"a": 1}\n'
        assert [p.name for p in dst.parent.iterdir()] == ["record.json"]

    def test_a_crash_mid_publish_keeps_the_previous_file(self, tmp_path, monkeypatch):
        import os

        dst = tmp_path / "record.json"
        dst.write_text("old", encoding="utf-8")

        def lost_mount(src, target):
            raise OSError("mount went away")

        monkeypatch.setattr(os, "replace", lost_mount)
        with pytest.raises(OSError, match="mount went away"):
            atomic_write_text(dst, "new")

        assert dst.read_text(encoding="utf-8") == "old"
        assert [p.name for p in tmp_path.iterdir()] == ["record.json"]


class TestPublishedFileMode:
    """A published temporary carries the mode a plain write would have given it.

    ``mkstemp`` creates 0600 regardless of the umask; ``summary.json`` and
    ``stage_result.json`` used to land owner-only beside the 0644 files
    ``Path.write_text`` and ``result_bundle.hashing._write_json`` produce.
    """

    @pytest.fixture(params=[0o022, 0o077, 0o002], ids=lambda mask: f"umask={mask:03o}")
    def umask(self, request):
        previous = os.umask(request.param)
        try:
            yield request.param
        finally:
            os.umask(previous)

    @staticmethod
    def _mode(path):
        return stat.S_IMODE(path.stat().st_mode)

    def test_atomic_write_text_matches_path_write_text(self, tmp_path, umask):
        plain = tmp_path / "plain.json"
        plain.write_text("{}", encoding="utf-8")

        atomic_write_text(tmp_path / "atomic.json", "{}")

        assert self._mode(tmp_path / "atomic.json") == self._mode(plain) == 0o666 & ~umask

    def test_atomic_copy_and_savez_match_a_plain_write(self, tmp_path, umask):
        plain = tmp_path / "plain.bin"
        plain.write_bytes(b"x")

        atomic_copy(plain, tmp_path / "copy.bin")
        atomic_savez(tmp_path / "arrays.npz", a=np.array([1]))

        assert self._mode(tmp_path / "copy.bin") == self._mode(plain)
        assert self._mode(tmp_path / "arrays.npz") == self._mode(plain)

    def test_a_replaced_file_takes_the_new_mode(self, tmp_path, umask):
        dst = tmp_path / "record.json"
        dst.write_text("old", encoding="utf-8")
        os.chmod(dst, 0o600)

        atomic_write_text(dst, "new")

        assert dst.read_text(encoding="utf-8") == "new"
        assert self._mode(dst) == 0o666 & ~umask
