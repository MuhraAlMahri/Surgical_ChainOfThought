"""
Atomic, manifest-backed caching utilities.
No more phantom "cached" files!
"""

import os
import json
import hashlib
import tempfile
import sqlite3
import torch


def sha1(obj) -> str:
    """Generate SHA1 hash from object."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def cache_key(img_path, model_id, res, extra=None):
    """Generate stable cache key from canonical inputs."""
    return sha1({
        "abspath": os.path.abspath(img_path),
        "model": model_id,
        "res": res,
        "extra": extra or {}
    })


def atomic_save_tensor(tensor, out_path):
    """Atomically save tensor (no partial/corrupt files)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(out_path), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(tensor, tmp)
        os.replace(tmp, out_path)  # Atomic on POSIX
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


class Manifest:
    """SQLite-backed manifest for tracking cached files."""
    
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS entries(
                k TEXT PRIMARY KEY,
                path TEXT,
                bytes INTEGER,
                mtime REAL
            )
        """)
        self.db.commit()
    
    def has(self, k, path):
        """
        Check if key is cached.
        CRITICAL: Verify BOTH manifest entry AND actual file on disk.
        """
        row = self.db.execute("SELECT bytes FROM entries WHERE k=?", (k,)).fetchone()
        if not row:
            return False
        # Trust disk, not just manifest
        return os.path.isfile(path) and os.path.getsize(path) == row[0]
    
    def put(self, k, path):
        """Record cache entry in manifest."""
        st = os.stat(path)
        self.db.execute(
            "INSERT OR REPLACE INTO entries(k,path,bytes,mtime) VALUES(?,?,?,?)",
            (k, path, st.st_size, st.st_mtime)
        )
        self.db.commit()
    
    def close(self):
        """Close database connection."""
        self.db.close()






