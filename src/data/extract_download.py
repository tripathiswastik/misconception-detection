"""Priority extractor: extracts CSVs and labelled/test images first, then unlabelled images."""
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DOWNLOADS_DIR = Path.home() / "Downloads"


def extract_priority():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    candidates = (
        list(DOWNLOADS_DIR.glob("*competition*.zip"))
        + list(DOWNLOADS_DIR.glob("*misconception*.zip"))
        + list(DATA_DIR.glob("*.zip"))
    )
    valid_zips = [z for z in candidates if z.stat().st_size > 1_000_000]
    
    if not valid_zips:
        print("[PriorityExtractor] No completed zip file found.")
        return False

    target_zip = sorted(valid_zips, key=lambda p: p.stat().st_size, reverse=True)[0]
    print(f"[PriorityExtractor] Opening archive: {target_zip.name} ({target_zip.stat().st_size / (1024**3):.2f} GB)")

    with zipfile.ZipFile(target_zip, "r") as zf:
        infolist = zf.infolist()
        print(f"[PriorityExtractor] Total entries in archive: {len(infolist)}")

        # Step 1: Extract all CSV files immediately
        csv_members = [m for m in infolist if m.filename.endswith(".csv")]
        print(f"[PriorityExtractor] Extracting {len(csv_members)} CSV files...")
        for m in csv_members:
            zf.extract(m, DATA_DIR)
        print("[PriorityExtractor] All CSV files extracted successfully!")

        # Step 2: Extract question diagrams, train images, and test images
        critical_image_members = [
            m for m in infolist
            if not m.is_dir() and (
                "images/train/" in m.filename
                or "images/test/" in m.filename
                or "images/questions/" in m.filename
            )
        ]
        print(f"[PriorityExtractor] Extracting {len(critical_image_members)} core images (train/test/questions)...")
        for idx, m in enumerate(critical_image_members, 1):
            zf.extract(m, DATA_DIR)
            if idx % 1000 == 0 or idx == len(critical_image_members):
                print(f"  Extracted {idx}/{len(critical_image_members)} core images...")

        print("[PriorityExtractor] Core images (train/test/questions) extracted successfully!")

        # Step 3: Extract unlabelled images
        unlabelled_members = [
            m for m in infolist
            if not m.is_dir() and "images/unlabelled/" in m.filename
        ]
        print(f"[PriorityExtractor] Extracting {len(unlabelled_members)} unlabelled images...")
        for idx, m in enumerate(unlabelled_members, 1):
            zf.extract(m, DATA_DIR)
            if idx % 5000 == 0 or idx == len(unlabelled_members):
                print(f"  Extracted {idx}/{len(unlabelled_members)} unlabelled images...")

    print("[PriorityExtractor] Full extraction complete!")
    return True


if __name__ == "__main__":
    extract_priority()
