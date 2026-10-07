"""Register future adapters in ADAPTERS. Each exposes metadata and
predict(image_bytes, wells), returning one record per well with well_id,
prediction (coma/awake/empty/unknown), confidence (0..1 or None), optional egg_count.
There are no fake model predictions in manual-review mode.
"""
import json
import math
import importlib.util
from backend.services.geometry import is_simple_polygon
from backend.services.keras_adapter import registered_adapters
ADAPTERS = registered_adapters()
def available_models():
    return [{'id': 'manual', 'name': 'Manual review', 'version': '1',
             'description': 'Annotate coma / awake / unknown; no automated predictions.',
             'task': 'coma', 'available': True}] + [dict(a.metadata, id=k, available=importlib.util.find_spec("tensorflow") is not None if hasattr(a,"load") else True) for k,a in ADAPTERS.items()]
def analyze(model_id, content, wells_json):
    if model_id not in ADAPTERS:
        raise ValueError('No trained adapter is available for this model. Use manual review.')
    wells = parse_wells(wells_json)
    ids = {w["id"] for w in wells}
    adapter = ADAPTERS[model_id]
    records = adapter.predict(content, wells)
    if not isinstance(records, list) or len(records) != len(wells):
        raise RuntimeError('Adapter must return exactly one record per well.')
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError('Adapter returned a malformed record.')
        well_id = record.get('well_id')
        confidence = record.get('confidence')
        if well_id not in ids or well_id in seen:
            raise RuntimeError('Adapter returned invalid well identities.')
        seen.add(well_id)
        if record.get('prediction') not in ('coma', 'awake', 'empty', 'unknown'):
            raise RuntimeError('Adapter returned an unsupported prediction.')
        if confidence is not None and (isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise RuntimeError('Adapter confidence must be null or a number from 0 to 1.')
        egg_count = record.get('egg_count')
        if egg_count is not None and (isinstance(egg_count, bool) or not isinstance(egg_count, int) or egg_count < 0):
            raise RuntimeError('Adapter egg count must be null or a nonnegative integer.')
    return {'model': dict(adapter.metadata, id=model_id), 'records': records}

def parse_wells(wells_json):
    try:
        wells = json.loads(wells_json)
        if not isinstance(wells, list) or not wells:
            raise ValueError()
        ids = set()
        for well in wells:
            if not isinstance(well, dict):
                raise ValueError()
            identity = well.get('id')
            points = well.get('points')
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError()
            ids.add(identity)
            if not is_simple_polygon(points):
                raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('Invalid well geometry.') from None
    return wells
