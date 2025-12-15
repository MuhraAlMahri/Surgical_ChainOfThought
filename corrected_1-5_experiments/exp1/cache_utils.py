"""
Robust caching utilities with stable keys and atomic writes.
"""

import os
import hashlib
import json
import tempfile
import torch


def cache_key(img_path, model_id, resolution_info, extra=None):
    """
    Generate stable, unique cache key from image path and processing parameters.
    Uses SHA1 of canonical inputs to avoid collisions.
    """
    meta = {
        "abspath": os.path.abspath(img_path),
        "model": model_id,
        "resolution": resolution_info,
        "extra": extra or {},
    }
    s = json.dumps(meta, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def atomic_save_tensor(data, out_path):
    """
    Atomically save tensor to avoid partial/corrupt files.
    Uses temp file + atomic rename.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(out_path), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(data, tmp)
        os.replace(tmp, out_path)  # Atomic on POSIX
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


def is_cached(path):
    """
    Verify file exists and has non-zero size.
    Catches partial writes and empty files.
    """
    return os.path.isfile(path) and os.path.getsize(path) > 0






