import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


PAGES_DIR = ROOT_DIR / "pages"
INGESTION_DIR = ROOT_DIR / "ingestion_program"
SCORING_DIR = ROOT_DIR / "scoring_program"
PHASE_DATA = ROOT_DIR / "dev_phase"

BUNDLE_FILES = [
    ROOT_DIR / "competition.yaml",
    ROOT_DIR / "logo.png",
    ROOT_DIR / "solution/submission.py",
]


if __name__ == "__main__":
    with zipfile.ZipFile("bundle.zip", mode="w") as bundle:
        for f in BUNDLE_FILES:
            f = f.relative_to(ROOT_DIR)
            print(f)
            bundle.write(f)
        for dirpath in [INGESTION_DIR, SCORING_DIR, PAGES_DIR, PHASE_DATA]:
            assert dirpath.exists(), (
                f"{dirpath} does not exist while it should. Make sure you "
                "followed all the instructions in the README before "
                "creating the bundle."
            )
            for f in dirpath.rglob("*"):
                if not f.is_file():
                    continue
                if f.name.startswith(".") or f.name.endswith(".pyc"):
                    continue
                f = f.relative_to(ROOT_DIR)
                print(f)
                bundle.write(f)
