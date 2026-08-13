"""Run the official MS-Transformer entry point unchanged on a modern stack.

Adds two inert attributes that old-torchvision code imports (their version guard
misfires on torchvision >= 0.20 — both symbols are imported but never called),
sets the torch>=2.6 legacy-pickle env var, then hands over to main.py via runpy.

Usage: python mst_wrap.py main.py <official args...>   (cwd must be the repo root)
"""
import os
import runpy
import sys
import torchvision.ops
import torchvision.ops.misc

if not hasattr(torchvision.ops, "_new_empty_tensor"):
    torchvision.ops._new_empty_tensor = lambda x, shape: x.new_empty(shape)
if not hasattr(torchvision.ops.misc, "_output_size"):
    torchvision.ops.misc._output_size = (
        lambda dim, input, size=None, scale_factor=None: [])
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
