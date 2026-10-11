"""Metadata-only pretraining split checks; never certifies asset provenance.

Rows contain sample_id, video_id, role (train/val/test), and optionally a
canonical media_sha256. Public encoder/cache provenance needs a separate audit.
Passing IDs/hashes does not rule out differently encoded near duplicates.
"""
import re


def _rows(rows):
    if not rows:
        raise ValueError('Nonempty metadata required')
    seen = set()
    for row in rows:
        if set(row) - {'sample_id', 'video_id', 'role', 'media_sha256'}:
            raise ValueError('Only label-free metadata fields allowed')
        if not all(isinstance(row.get(k), str) and row[k] for k in ('sample_id', 'video_id', 'role')):
            raise ValueError('Explicit sample/video/role metadata required')
        if row['role'] not in {'train', 'val', 'test'} or row['sample_id'] in seen:
            raise ValueError('Invalid role or duplicate sample ID')
        seen.add(row['sample_id'])
        h = row.get('media_sha256')
        if h is not None and not re.fullmatch('[0-9a-f]{64}', h):
            raise ValueError('Invalid canonical media SHA256')
    return rows


def check_metadata(target_rows, external_rows, target_pretraining_sample_ids):
    target = _rows(target_rows)
    external = _rows(external_rows)
    target_by_id = {r['sample_id']: r for r in target}
    chosen = list(target_pretraining_sample_ids)
    if len(chosen) != len(set(chosen)):
        raise ValueError('Duplicate target pretraining IDs')
    if any(s not in target_by_id or target_by_id[s]['role'] != 'train' for s in chosen):
        raise ValueError('Target pretraining must use official TRAIN inputs only')
    videos = {role: {r['video_id'] for r in target if r['role'] == role} for role in ('train', 'val', 'test')}
    if any(videos[a] & videos[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Official target roles overlap by video')
    if any(r['role'] != 'train' for r in external):
        raise ValueError('External manifest must be restricted to its official TRAIN inputs')
    if {r['video_id'] for r in external} & set.union(*videos.values()):
        raise ValueError('Cross-dataset canonical video overlap: exclude before training')
    th = {r['media_sha256'] for r in target if r.get('media_sha256')}
    eh = {r['media_sha256'] for r in external if r.get('media_sha256')}
    if th & eh:
        raise ValueError('Cross-dataset canonical media hash overlap')
    return dict(status='DECLARED_METADATA_CHECK_ONLY',
                target_pretraining_rows=len(chosen), external_pretraining_rows=len(external),
                asset_or_encoder_provenance_verified=False, actual_training_executed=False,
                canonical_media_hash_coverage_complete=all(r.get('media_sha256') for r in target + external),
                near_duplicate_or_speaker_overlap_audit='NOT_PERFORMED',
                full_pretraining_qualification=False)
