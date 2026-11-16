from pathlib import Path
from shutil import rmtree

from . import snli_processing, snli_architectures
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
    snli, fold_ids = snli_processing.get_snli_dataset()
    snli_processing.print_snli_summary(snli, fold_ids)
    embedding_model = snli_processing.get_embedding_model()
    snli_processing.print_embedding_summary(embedding_model)
    snli_architectures.main(snli, embedding_model)


if __name__ == "__main__":
    main()
