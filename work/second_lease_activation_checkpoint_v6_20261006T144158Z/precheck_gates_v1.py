"""Pure evidence-format checks, not replacement for trusted human authorization."""
from datetime import datetime, timezone, timedelta

UUIDS = {'a': 'GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7',
         'b': 'GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067',
         'c': 'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}

def timestamp(value):
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise PermissionError('TIMESTAMP_NEEDS_TIMEZONE')
    return t.astimezone(timezone.utc)

def validate_precheck_evidence(e, now, runtime_plan_sha):
    # Caller must have verified provenance outside this JSON checker. A fixture,
    # stale automation text, or suggested lease date is not provider evidence.
    if e.get('scope') != 'NEW_MINIMAL_FIXED_FOLD0_GPU_PRECHECK_ONLY':
        raise PermissionError('PRECHECK_SCOPE')
    if not e.get('trusted_human_or_provider_provenance_verified', False):
        raise PermissionError('TRUSTED_PROVENANCE_NOT_VERIFIED')
    if not e.get('platform_lease_confirmed', False):
        raise PermissionError('ESTIMATED_LEASE_IS_NOT_CONFIRMED')
    if timestamp(e['lease_end_utc']) - now < timedelta(seconds=7200+1200):
        raise PermissionError('PRECHECK_PLUS_TWO_HOURS_PRESERVATION_RESERVE')
    age = (now-timestamp(e['queried_actual_utc'])).total_seconds()
    if not 0 <= age <= 300:
        raise PermissionError('FRESH_READONLY_EVIDENCE_REQUIRED')
    if e.get('node') not in UUIDS or e.get('gpu_uuid') != UUIDS[e['node']]:
        raise PermissionError('AUTHORIZED_GPU_UUID_MISMATCH')
    if e.get('compute_processes') != [] or not isinstance(e.get('python_full_argv'), list):
        raise PermissionError('COMPUTE_OR_COMPLETE_ARGV_NOT_VERIFIED')
    if not e.get('host_identity_and_credentials_verified', False):
        raise PermissionError('HOST_IDENTITY_OR_CREDENTIALS_NOT_VERIFIED')
    if e.get('runtime_plan_sha256') != runtime_plan_sha or not e.get('assets_source_complete_space_verified', False):
        raise PermissionError('SOURCE_ASSET_COMPLETE_SPACE_EVIDENCE_MISSING')
    # Two new full states plus archives/receipts, without deleting old files.
    if e.get('remote_free_bytes', 0) < 4*1024**3 or e.get('permanent_D_free_bytes', 0) < 6*1024**3:
        raise PermissionError('COMPLETE_INITIAL_AND_PRECHECK_STATE_PRESERVATION_SPACE')
    return {'scope': e['scope'], 'node': e['node'], 'gpu_uuid': e['gpu_uuid'],
            'runtime_plan_sha256': runtime_plan_sha, 'precheck_only': True,
            'formal100_or_outer_authorized': False}

def validate_saved_state_metadata(meta, expected_plan_sha, expected_scope, expected_steps):
    if meta.get('format') != 'MINIMAL_FIXED_FULL_MODEL_STATE_V1':
        raise ValueError('FULL_STATE_FORMAT')
    if meta.get('runtime_plan_sha256') != expected_plan_sha or meta.get('fold') != 0 or meta.get('seed') != 91819:
        raise ValueError('STATE_SOURCE_ROLE_INITIALIZATION_BOUNDARY')
    if meta.get('scope') != expected_scope or meta.get('optimizer_steps') != expected_steps:
        raise ValueError('CLEAN_INITIAL_AND_PRECHECK_STATE_MUST_NOT_BE_SUBSTITUTED')
    for key in ('state_sha256', 'fit_statistics_sha256'):
        h = meta.get(key, '')
        if len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('STATE_OR_FIT_STATISTICS_SHA_FORMAT')
    return True
