"""Prepare crystal JEPA train/valid/test manifests from the existing LMDB cache.

This source-specific split step has no upstream equivalent. Crystal graph
extraction uses Meta's all-atom-diffusion-transformer code during loading.
"""
from lvebcsp.data.crystal_splits import main

if __name__ == "__main__":
    main()
