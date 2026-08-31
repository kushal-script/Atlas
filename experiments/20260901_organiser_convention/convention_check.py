"""Build a pair with the organisers' own transform and ask what we report.

  p_search = (1/z) * R(theta) * (p_canvas - c_canvas) + c_search
  R(theta) = [[cos t,  sin t],
              [-sin t, cos t]]            t = radians(theta)

Reference is 1000x1000 at 1.0 nm/px cut from the canvas unrotated; search is
1000x1000 at z nm/px. Ground truth x,y is the reference centre pushed through
the transform above."""
import sys, numpy as np, cv2
sys.path.insert(0, "src")
from drift_sense.localize import MatchConfig, locate

def R(theta_deg):
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]], float)

def build(canvas, c_canvas, z, theta, out=1000):
    """Render the search image by inverting the organisers' forward map."""
    c_search = np.array([(out - 1) / 2.0, (out - 1) / 2.0])
    Rinv = np.linalg.inv(R(theta))
    ys, xs = np.mgrid[0:out, 0:out].astype(np.float64)
    d = np.stack([xs.ravel() - c_search[0], ys.ravel() - c_search[1]])
    src = (Rinv @ d) * z + np.array(c_canvas).reshape(2, 1)
    mx = src[0].reshape(out, out).astype(np.float32)
    my = src[1].reshape(out, out).astype(np.float32)
    # supersample the canvas read to avoid aliasing at z of 8 to 12
    return cv2.remap(canvas, mx, my, cv2.INTER_AREA if False else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

def gt_xy(p_canvas, c_canvas, z, theta, out=1000):
    c_search = np.array([(out - 1) / 2.0, (out - 1) / 2.0])
    d = np.array(p_canvas, float) - np.array(c_canvas, float)
    return tuple((R(theta) @ d) / z + c_search)

rng = np.random.default_rng(4242)
N = 16000
canvas = rng.normal(128, 30, (N, N)).astype(np.float32)
# aperiodic landmarks so the match is unique
for _ in range(60):
    cx, cy = rng.integers(2000, N - 2000, 2)
    w, h = rng.integers(200, 900, 2)
    canvas[cy:cy + h, cx:cx + w] += rng.uniform(30, 80)
canvas = np.clip(cv2.GaussianBlur(canvas, (0, 0), 3.0), 0, 255)

print(f"  {'z':>6}{'theta in':>10}{'our theta':>11}{'our scale':>11}{'err px':>9}  verdict")
bad = 0
for z, theta in ((10.0, 0.0), (9.0, 3.5), (11.0, -3.5), (8.0, 5.0), (12.0, -5.0)):
    c_canvas = (N / 2.0, N / 2.0)
    search = build(canvas, c_canvas, z, theta)
    # reference: unrotated 1000x1000 crop at 1 nm/px, centred on a point that
    # lands well inside the search frame
    p_canvas = (N / 2.0 + 900.0, N / 2.0 - 600.0)
    x0, y0 = int(p_canvas[0] - 500), int(p_canvas[1] - 500)
    ref = canvas[y0:y0 + 1000, x0:x0 + 1000]
    gx, gy = gt_xy(p_canvas, c_canvas, z, theta)
    x, y, d, _ = locate(np.clip(ref, 0, 255).astype(np.uint8),
                        np.clip(search, 0, 255).astype(np.uint8), MatchConfig())
    cfg = MatchConfig()
    rep_theta = cfg.theta_report_sign * float(d["theta_deg"])
    rep_scale = float(d["scale"]) * cfg.zoom
    err = float(np.hypot(x - gx, y - gy))
    ok = abs(rep_theta - theta) < 1.0 and err < 5.0
    bad += (not ok)
    print(f"  {z:6.1f}{theta:10.2f}{rep_theta:11.2f}{rep_scale:11.3f}{err:9.2f}  "
          f"{'OK' if ok else 'MISMATCH'}")
print(f"\n  {'ALL CONSISTENT' if bad == 0 else str(bad) + ' MISMATCHES'} with the published convention")
