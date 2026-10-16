"""Lazy Pillow loader for FlyScope datasets (including old pre-cropped exports).

Unzip ALL archive parts into one directory. No per-frame crop files are required.
Split by annotation['split_group'], never by random adjacent frames.
"""
import json
from pathlib import Path
import runpy


def samples(root, include_review=False):
    from PIL import Image, ImageDraw
    root = Path(root).resolve()
    manifest = json.loads((root / 'annotations.json').read_text())
    if manifest.get('completion_marker'):
        if manifest['completion_marker'] != 'dataset-complete.json' or not (root / 'dataset-complete.json').is_file():
            raise ValueError('Incomplete dataset. Extract every ZIP part or export again.')
    if manifest.get('geometry_format') == 'shared-well-boundaries-v1':
        hydrate = runpy.run_path(str(Path(__file__).with_name('dataset_format.py')))['hydrate_manifest']
        hydrate(manifest)
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
        relative = annotation.get('crop_path') or annotation['image_path']
        source_path = (root / relative).resolve()
        if not source_path.is_relative_to(root):
            raise ValueError('Image path leaves dataset directory.')
        with Image.open(source_path) as source:
            if annotation.get('crop_path'):
                rgba = source.convert('RGBA')
                image = Image.new('RGB', rgba.size, 'white')
                image.paste(rgba, mask=rgba.getchannel('A'))
            else:
                import math
                points = [(x * source.width, y * source.height) for x, y in annotation['boundary_normalized']]
                left, top = max(0, math.floor(min(x for x, _ in points))), max(0, math.floor(min(y for _, y in points)))
                right, bottom = min(source.width, math.ceil(max(x for x, _ in points))), min(source.height, math.ceil(max(y for _, y in points)))
                if left >= right or top >= bottom:
                    raise ValueError('Empty annotation boundary.')
                crop = source.convert('RGB').crop((left, top, right, bottom))
                mask = Image.new('L', crop.size)
                ImageDraw.Draw(mask).polygon([(x-left, y-top) for x,y in points], fill=255)
                image = Image.new('RGB', crop.size, 'white')
                image.paste(crop, mask=mask)
        yield {'image': image, 'target': target, 'annotation': annotation}


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    print(f'{sum(1 for _ in samples(root))} eligible manual training samples. Split by experiment group.')
