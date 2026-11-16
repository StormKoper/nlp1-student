from pathlib import Path
from shutil import rmtree

from . import snli_processing, snli_tokenization, snli_architectures
from .snli_config import ROOT


def clear_data_dir(path: Path) -> None:
    if not path.exists():
        return

    data_root = path.resolve()
    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            target = item.resolve()
            if not target.is_relative_to(data_root):
                raise ValueError(f"Refusing to clear a directory outside {data_root}: {target}")
            rmtree(target)
        else:
            item.unlink()


def main() -> None:
    clear_data_dir(ROOT / "data")
    snli_processing.main()
    snli_tokenization.main()
    snli_architectures.main()


if __name__ == "__main__":
    main()
