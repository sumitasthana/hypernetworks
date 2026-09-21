"""Step 1: read Tiny ImageNet without changing its files."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "tiny-imagenet-200"
URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


def prepare_data(root=DEFAULT_ROOT, download=False):
    """Return the extracted dataset directory; download only when requested."""
    root = Path(root).expanduser().resolve()
    if not root.exists() and download:
        if root.name != "tiny-imagenet-200":
            raise ValueError("For download, root must end in tiny-imagenet-200")
        download_and_extract_archive(URL, str(root.parent))
    for name in ("wnids.txt", "words.txt", "train", "val/val_annotations.txt"):
        if not (root / name).exists():
            raise FileNotFoundError(
                f"Missing {root / name}. Supply the extracted dataset root, "
                "or use download=True for a new dataset."
            )
    return root


class TinyImageNet(Dataset):
    """Return (RGB image, integer label) for the train or validation split.

    Both splits use sorted wnids.txt IDs for exactly the same label mapping.
    A transform can turn the PIL image into a tensor. Images load on demand.
    The public test split has no labels, so it is deliberately not accepted.
    """

    def __init__(self, root=DEFAULT_ROOT, split="train", transform=None):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'; test labels are unavailable")
        self.root = prepare_data(root)
        self.split = split
        self.transform = transform
        self.classes = sorted((self.root / "wnids.txt").read_text().split())
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("wnids.txt must contain unique class IDs")
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.classes)}
        words = {}
        for line in (self.root / "words.txt").read_text(encoding="utf-8").splitlines():
            wnid, description = line.split("\t", 1)
            words[wnid] = description
        self.class_names = [words.get(wnid, wnid) for wnid in self.classes]
        self.samples = []

        if split == "train":
            for wnid in self.classes:
                folder = self.root / "train" / wnid
                # The official box file is also an image manifest. Prefer it to
                # a directory scan, which can race with another loader moving files.
                manifest = folder / f"{wnid}_boxes.txt"
                if manifest.is_file():
                    names = sorted(line.split()[0] for line in manifest.read_text().splitlines() if line.strip())
                    if len(names) != len(set(names)):
                        raise ValueError(f"Duplicate training images in {manifest}")
                    paths = [folder / "images" / name for name in names]
                    paths = [p if p.is_file() else folder / p.name for p in paths]
                    missing = next((p for p in paths if not p.is_file()
                                    and not (folder / p.name).is_file()
                                    and not (folder / "images" / p.name).is_file()), None)
                    if missing is not None:
                        raise FileNotFoundError(f"Missing training image: {missing}")
                else:
                    paths = sorted(folder.rglob("*.JPEG"))
                if not paths:
                    raise FileNotFoundError(f"No training images in {folder}")
                self.samples.extend((p, self.class_to_idx[wnid]) for p in paths)
        else:
            annotations = self.root / "val" / "val_annotations.txt"
            seen = set()
            for line in annotations.read_text().splitlines():
                filename, wnid, *_ = line.split("\t")
                if filename in seen:
                    raise ValueError(f"Duplicate validation image: {filename}")
                seen.add(filename)
                candidates = [
                    self.root / "val" / "images" / filename,
                    self.root / "val" / wnid / filename,
                    self.root / "val" / wnid / "images" / filename,
                ]
                path = next((p for p in candidates if p.is_file()), None)
                if path is None:
                    raise FileNotFoundError(f"Missing validation image: {filename}")
                self.samples.append((path, self.class_to_idx[wnid]))
        if not self.samples:
            raise ValueError(f"Empty {split} split")
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def image_path(self, index):
        """Resolve an indexed image even if another loader flattened its folder."""
        path, target = self.samples[index]
        if path.is_file():
            return path
        folder = self.root / self.split / self.classes[target]
        for candidate in (folder / path.name, folder / "images" / path.name):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Missing indexed image: {path}")

    def __getitem__(self, index):
        _, target = self.samples[index]
        path = self.image_path(index)
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target
