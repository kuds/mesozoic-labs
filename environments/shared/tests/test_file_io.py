"""Tests for atomic file-write helpers."""

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
