"""Expand compact dataset geometry for legacy and current model consumers.

This standard-library-only module is also included in standalone dataset exports.
Point lists are shared in memory, never copied for every frame/annotation.
"""


def hydrate_manifest(manifest):
    # Browser exports use frame_interval_seconds; keep one internal spelling.
    if (
        "interval" not in manifest
        and manifest.get("frame_interval_seconds") is not None
    ):
        manifest["interval"] = manifest["frame_interval_seconds"]
    if manifest.get("geometry_format") != "shared-well-boundaries-v1":
        return manifest
    images = {im["frame_index"]: im for im in manifest["images"]}
    wells = {w["well_uid"]: w for w in manifest["wells"]}
    if len(wells) != len(manifest["wells"]):
        raise ValueError("Duplicate well IDs in dataset.")
    for w in wells.values():
        active = None if w.get("active_frames") is None else set(w["active_frames"])
        reviews = set(w.get("review_frames", []))
        w["frames"] = [
            {
                "frame_index": i,
                "points": w.get("overrides", {}).get(str(i), w["points"]),
                "active": True,
                "partial": w.get("partial_frames", {}).get(
                    str(i), w.get("partial", False)
                ),
                "boundary_reviewed": not w.get("review_required", False)
                and i not in reviews,
            }
            for i in images
            if active is None or i in active
        ]
    geometries = {
        uid: {g["frame_index"]: g for g in w["frames"]} for uid, w in wells.items()
    }
    for a in manifest["annotations"]:
        try:
            im, w = images[a["frame_index"]], wells[a["well_uid"]]
            points = geometries[a["well_uid"]][a["frame_index"]]["points"]
        except KeyError as exc:
            raise ValueError(
                "Annotation references missing image or inactive well."
            ) from exc
        a.update(
            image_path=im["path"],
            image_width=im["width"],
            image_height=im["height"],
            boundary_normalized=points,
            frame_number=im["frame_number"],
            filename=im["original_filename"],
            well_id=w["well_id"],
            split_group=manifest["split_group"],
            species=w.get("species"),
            task=manifest["task"],
            annotation_unit=w.get("annotation_unit", "well"),
        )
    return manifest
