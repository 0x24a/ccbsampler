from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CACHE_EXT = ".hifi.npz"


class CacheManager:
    def __init__(self, max_concurrent_generation: int = 1) -> None:
        # Limit concurrent mel extraction to avoid OOM when many cache misses hit simultaneously
        self._generate_sem = asyncio.Semaphore(max_concurrent_generation)
        self._path_locks: dict[Path, asyncio.Lock] = {}

    def _lock_for(self, path: Path) -> asyncio.Lock:
        # Serving one asyncio loop, so a dict lookup is safe without extra sync
        return self._path_locks.setdefault(path, asyncio.Lock())

    @staticmethod
    def cache_path(wav_path: Path, flag_suffix: str) -> Path:
        name = f"{wav_path.stem}_{flag_suffix}" if flag_suffix else wav_path.stem
        return wav_path.with_name(name + CACHE_EXT)

    async def get_or_generate(
        self,
        wav_path: Path,
        flag_suffix: str,
        force: bool,
        generate_fn,  # sync callable () -> dict[str, np.ndarray]
    ) -> dict:
        cache_path = self.cache_path(wav_path, flag_suffix)

        # Check cache first without any lock (fast path for hits)
        if not force and cache_path.exists():
            try:
                features = await asyncio.to_thread(
                    lambda: dict(np.load(str(cache_path)))
                )
                logger.info("Cache hit: %s", cache_path.name)
                return features
            except (EOFError, OSError, ValueError):
                pass  # corrupted — fall through to regenerate

        # Cache miss or force: serialise generation to avoid OOM,
        # then re-check under the per-path lock in case another request
        # already generated it while we were waiting.
        async with self._generate_sem:
            async with self._lock_for(cache_path):
                if not force and cache_path.exists():
                    try:
                        features = await asyncio.to_thread(
                            lambda: dict(np.load(str(cache_path)))
                        )
                        logger.info("Cache hit (after lock): %s", cache_path.name)
                        return features
                    except (EOFError, OSError, ValueError) as e:
                        logger.warning(
                            "Corrupted cache %s (%s), regenerating", cache_path.name, e
                        )
                        try:
                            os.remove(cache_path)
                        except OSError:
                            pass

                logger.info("Cache miss: %s, generating", cache_path.name)
                return await self._generate(cache_path, generate_fn)

    @staticmethod
    async def _generate(cache_path: Path, generate_fn) -> dict:
        features: dict = await asyncio.to_thread(generate_fn)
        tmp_npz = cache_path.with_name(cache_path.name + ".tmp")
        try:
            await asyncio.to_thread(
                lambda: np.savez_compressed(str(tmp_npz), **features)
            )
            actual_tmp = tmp_npz.with_name(tmp_npz.name + ".npz")
            await asyncio.to_thread(os.replace, str(actual_tmp), str(cache_path))
            logger.info("Cache saved: %s", cache_path.name)
        except Exception:
            for p in (tmp_npz, tmp_npz.with_name(tmp_npz.name + ".npz")):
                if p.exists():
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            raise

        return features
