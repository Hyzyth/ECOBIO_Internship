"""Framework-neutral loader for an exported FlyScope training archive.

Unzip the archive, install Pillow, then import samples(path) into your trainer.
Split by annotation['split_group'], never by random adjacent frames.
"""
import json
from pathlib import Path


def samples(root, include_review=False):
    from PIL import Image
    root = Path(root).resolve()
    manifest = json.loads((root / 'annotations.json').read_text())
    for annotation in manifest['annotations']:
        if not include_review and not annotation['eligible_for_training']:
            continue
        label = annotation['label']
        if manifest['task'] == 'ccrt':
            if label not in manifest['classes']:
                continue
            target = manifest['classes'][label]
        else:
            target = label
        crop_path = (root / annotation['crop_path']).resolve()
        if not crop_path.is_relative_to(root):
            raise ValueError('Crop path leaves dataset directory.')
        with Image.open(crop_path) as source:
            rgba = source.convert('RGBA')
            image = Image.new('RGB', rgba.size, 'black')
            image.paste(rgba, mask=rgba.getchannel('A'))
        yield {'image': image, 'target': target, 'annotation': annotation}


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    count = sum(1 for _ in samples(root))
    print(f'{count} eligible manual training samples. Split by experiment group.')
