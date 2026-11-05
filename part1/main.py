from pathlib import Path
from shutil import rmtree

import snli_processing
import snli_representations
import snli_tokenizer


def clear_data_dir(path: Path) -> None:
    for dir in path.iterdir():
        rmtree(dir)

def main() -> None:
    clear_data_dir(Path("part1/data"))
    snli_processing.main()
    snli_tokenizer.main()
    snli_representations.main()

if __name__ == "__main__":
    main()
