from pathlib import Path


# Resolve project paths relative to the package location.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_FOLDER_RAW = str(_PROJECT_ROOT / "data")
PATH_FOLDER_IMAGE_RAW = str(Path(PATH_FOLDER_RAW) / "image")
PATH_FOLDER_IMAGE_TEST = str(Path(PATH_FOLDER_RAW) / "small")
PATH_FOLDER_TABLE_RAW = str(Path(PATH_FOLDER_RAW) / "table")


CLASS_NAMES = [
    "dog",
    "spider",
    "cat",
    "sheep",
    "elephant",
    "chicken",
    "butterfly",
    "cow",
    "squirrel",
    "horse",
]
CLASS_INDEX = {
    "dog": 1,
    "spider": 2,
    "cat": 3,
    "sheep": 4,
    "elephant": 5,
    "chicken": 6,
    "butterfly": 7,
    "cow": 8,
    "squirrel": 9,
    "horse": 10,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

DEFAULT_SIZE = (128,128)