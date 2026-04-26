from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import numpy as np
from filelock import FileLock

logger = logging.getLogger(__name__)

CACHE_EXT = ".hifi.npz"
LOCK_TIMEOUT = 60


class AsyncFileLock:
    def __init__(self, path: Path) -> None:
        self._lock = FileLock(str(path) + ".lock", timeout=LOCK_TIMEOUT)

    async def __aenter__(self) -> AsyncFileLock:
        await asyncio.to_thread(self._lock.acquire)
        return self

    async def __aexit__(self, *_: object) -> None:
        self._lock.release()


class CacheManager:
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

        async with AsyncFileLock(cache_path):
            if force:
                logger.info("G flag set, forcing feature regeneration")
                return await self._generate(cache_path, generate_fn)

            if cache_path.exists():
                try:
                    features = await asyncio.to_thread(
                        lambda: dict(np.load(str(cache_path)))
                    )
                    logger.info("Cache hit: %s", cache_path.name)
                    return features
                except (EOFError, OSError, ValueError) as e:
                    logger.warning("Corrupted cache %s (%s), regenerating", cache_path.name, e)
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
