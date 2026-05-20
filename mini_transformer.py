import os

from minitransformer.cli import main


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    main(default_model="v1")