"""Fixed low-capacity controls. Fit only complete video-isolated OOF packets.

No label-optimal weight estimated on an evaluation role may enter prediction.
Lineage validation is a guard, not evidence that externally supplied SHA is real.
"""
import numpy as np
from oof_selector import check_lineage, weights, video


def packet(p0, p1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if p0.ndim != 1 or p1.shape != p0.shape or len(p0) == 0:
        raise ValueError('Nonempty same-row scalar predictions required')
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError('Finite predictions required')
    return p0, p1


def error_geometry(y, p0, p1, sample_weight):
    """Descriptive role-specific statistics; never fits deployable controls."""
    p0, p1 = packet(p0, p1)
    y, w = np.asarray(y, float), np.asarray(sample_weight, float)
    if y.shape != p0.shape or w.shape != p0.shape or not np.isfinite(y).all():
        raise ValueError('Same-row labels and weights required for diagnostic')
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Nonnegative population weights required')
    w = w / w.sum()
    e0, e1, delta = y - p0, y - p1, p1 - p0
    a, b, c = float(w @ (e0*e0)), float(w @ (e1*e1)), float(w @ (e0*e1))
    D = float(w @ (delta*delta))
    mean0, mean1 = float(w @ e0), float(w @ e1)
    v0, v1 = float(w @ ((e0-mean0)**2)), float(w @ ((e1-mean1)**2))
    pearson = float(w @ ((e0-mean0)*(e1-mean1)) / np.sqrt(v0*v1)) if v0*v1 > 0 else None
    oracle = float(w @ (e0*delta) / D) if D > 0 else 0.
    return dict(a_MSE=a, b_MSE=b, c_second_crossmoment=c, D=D,
                bias0=mean0, bias1=mean1, Pearson_error_corr=pearson,
                uncentered_error_cosine=c/np.sqrt(a*b) if a*b > 0 else None,
                second_moment_threshold=np.sqrt(a/b) if b > 0 else None,
                diagnostic_only_label_optimal_w=oracle,
                diagnostic_only_convex_w=float(np.clip(oracle, 0, 1)),
                deployable_parameters=False)


class LinearControls:
    """Three predeclared OOF-trained controls; no sweeps or evaluation labels.

    1. p0 + w*(p1-p0), w any real scalar.
    2. intercept + slope*p0.
    3. calibrated p0 plus the delta component linearly orthogonal to [1,p0].

    Third arm is nested relative to the calibration-only arm. Lstsq uses
    NumPy's documented default numerical rank; rank-deficient arms fall back.
    """
    def fit(self, y, p0, p1, ids, official_train_ids, calibration_ids, lineage):
        check_lineage(ids, official_train_ids, calibration_ids, lineage)
        p0, p1 = packet(p0, p1)
        y = np.asarray(y, float)
        if y.shape != p0.shape or len(ids) != len(y) or not np.isfinite(y).all():
            raise ValueError('Finite same-row genuine OOF arrays required')
        w = weights(ids)
        d = p1-p0
        den = float(w @ (d*d))
        self.blend_w = float(w @ ((y-p0)*d) / den) if den > 0 else 0.
        A = np.column_stack((np.ones(len(p0)), p0))
        Aw = A*np.sqrt(w)[:,None]
        self.cal, _, self.rank, self.singular = np.linalg.lstsq(Aw, y*np.sqrt(w), rcond=None)
        self.delta_projection = np.linalg.lstsq(Aw, d*np.sqrt(w), rcond=None)[0]
        i = d-A@self.delta_projection
        # Rank check distinguishes a genuinely new direction from exact affine d.
        _, _, rank3, singular3 = np.linalg.lstsq(np.column_stack((Aw, i*np.sqrt(w))), y*np.sqrt(w), rcond=None)
        self.delta_gamma = float(w @ ((y-A@self.cal)*i) / (w@(i*i))) if rank3 > self.rank else 0.
        self.rank3, self.singular3 = rank3, singular3
        self.fit_ids = list(map(str, ids))
        self.fit_videos = {video(s) for s in ids}
        self.calibration_ids = list(map(str, calibration_ids))
        self.fitted = True
        return self

    def predict(self, p0, p1):
        if not getattr(self, 'fitted', False):
            raise ValueError('Controls require genuine OOF fit before evaluation')
        p0, p1 = packet(p0, p1)
        A = np.column_stack((np.ones(len(p0)), p0))
        d = p1-p0
        q = A@self.cal
        return dict(OOF_unconstrained_blend=p0+self.blend_w*d,
                    OOF_calibration_only=q,
                    OOF_calibration_plus_delta=q+self.delta_gamma*(d-A@self.delta_projection))

    def state(self):
        if not getattr(self, 'fitted', False):
            raise ValueError('No fitted state')
        return dict(blend_w=self.blend_w, calibration_intercept_slope=self.cal.tolist(),
                    delta_projection_intercept_slope=self.delta_projection.tolist(),
                    delta_gamma=self.delta_gamma, calibration_rank=int(self.rank),
                    combined_rank=int(self.rank3), calibration_singular=self.singular.tolist(),
                    combined_singular=self.singular3.tolist(), fit_ids=self.fit_ids,
                    calibration_ids=self.calibration_ids, risk_guarantee=False)
