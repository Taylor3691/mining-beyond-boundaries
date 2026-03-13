import importlib.util
from pathlib import Path

_module_path = Path(__file__).parent / "file.hihi.py"
_spec = importlib.util.spec_from_file_location("utils.file_hihi", _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_images = _mod.load_images
load_table = _mod.load_table
save_images = _mod.save_images
save_table = _mod.save_table
