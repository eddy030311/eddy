"""In-between ("transition branch") poses.

Each pose with an outstretched arm is split into two layers:
  * the arm (cut at a joint, rotates about that joint)
  * the body (arm removed, the hole filled from the aligned 'think' pose)
An in-between pose = body morph + arm rotated/foreshortened toward the next pose.
"""
import numpy as np, cv2
from PIL import Image
import rig

# arm polygons in pose-local pixels, and the joint the arm rotates about
ARMS = {
    'point': dict(poly=[(0, 212), (40, 212), (78, 236), (120, 238), (178, 243), (240, 248), (266, 262), (272, 300),
                        (262, 352), (226, 394), (165, 398), (118, 388), (68, 374), (42, 334), (16, 324), (0, 304)],
                  pivot=(252, 305)),
    'offer': dict(poly=[(26, 338), (108, 314), (192, 340), (228, 366), (252, 392), (292, 408), (346, 438), (362, 500),
                        (344, 562), (290, 568), (238, 524), (188, 482), (138, 452), (126, 420), (58, 398), (26, 372)],
                  pivot=(318, 528)),
}
CW, CH, CMX, CMY = 1000, 1900, 500, 250

def norm_affine(name):
    P = rig.POSES[name]; k = 64.0 / P['io']
    return np.float32([[k, 0, CMX - k * P['mid'][0]], [0, k, CMY - k * P['mid'][1]]])

def inv(M):
    return cv2.invertAffineTransform(M)

_layers = {}
def layers(name):
    """(body_rgba, arm_rgba, arm_mask) in pose-local space, premultiplied."""
    if name in _layers: return _layers[name]
    src = rig.load(name); h, w = src.shape[:2]
    A = ARMS[name]
    m = np.zeros((h, w), np.float32)
    cv2.fillPoly(m, [np.int32(A['poly'])], 1.0, cv2.LINE_AA)
    # cut at the joint with a soft edge so the arm blends into the shoulder / elbow
    px, py = A['pivot']
    gx, gy = rig.grid(w, h)
    m_soft = cv2.GaussianBlur(m, (0, 0), 2.0)
    arm = src * m_soft[..., None]
    # hole fill: the 'think' pose aligned into this pose's space, used where the arm was
    M = inv(norm_affine(name)); Mt = norm_affine('think')
    full = np.vstack([Mt, [0, 0, 1]]); Mi = np.vstack([M, [0, 0, 1]])
    think_here = cv2.warpAffine(rig.load('think'), (Mi @ full)[:2].astype(np.float32), (w, h), flags=cv2.INTER_LINEAR)
    hole = cv2.GaussianBlur(cv2.dilate(m, np.ones((7, 7), np.uint8)), (0, 0), 3.0)[..., None]
    body = src * (1 - hole) + think_here * hole
    _layers[name] = (body.astype(np.float32), arm.astype(np.float32), m_soft)
    return _layers[name]

def arm_transform(name, angle, fore):
    """Affine (pose-local) rotating the arm about its joint, foreshortened along its length by `fore`."""
    A = ARMS[name]; px, py = A['pivot']
    cx, cy = np.mean(np.array(A['poly'], float), axis=0)
    ax = np.arctan2(cy - py, cx - px)
    # scale along the arm axis, about the pivot
    c, s = np.cos(ax), np.sin(ax)
    Rm = np.array([[c, -s], [s, c]]); S = np.diag([fore, 1.0])
    L = Rm @ S @ Rm.T
    a = np.deg2rad(-angle)  # positive angle = counter-clockwise on screen
    R2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    T = R2 @ L
    p = np.array([px, py])
    return np.hstack([T, (p - T @ p)[:, None]]).astype(np.float32)

def posed(name, angle, fore, body=None):
    """Pose `name` with its arm rotated: returns pose-local premultiplied RGBA."""
    b, arm, _ = layers(name)
    if body is not None: b = body
    h, w = b.shape[:2]
    a2 = cv2.warpAffine(arm, arm_transform(name, angle, fore), (w, h), flags=cv2.INTER_LINEAR)
    return b * (1 - a2[..., 3:4]) + a2

# Transition branches: key frames (pose, arm angle, foreshortening) the move passes through
BRANCHES = {
    ('offer', 'point'): [('offer', 0, 1.0), ('offer', -38, 0.72), ('point', 42, 0.62), ('point', 0, 1.0)],
    ('point', 'think'): [('point', 0, 1.0), ('point', -30, 0.78), ('point', -62, 0.5), ('think', None, None)],
}

def sheet(path):
    rows = []
    for key, keys in BRANCHES.items():
        tiles = []
        for (name, ang, fore) in keys:
            if ang is None:
                img = rig.load(name)
            else:
                img = posed(name, ang, fore)
            n = cv2.warpAffine(img, norm_affine(name), (CW, CH))[:1150, 60:940]
            rgb = n[..., :3] + (1 - n[..., 3:])
            tiles.append(cv2.resize((np.clip(rgb, 0, 1) * 255).astype(np.uint8), (440, 575), interpolation=cv2.INTER_AREA))
        rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)

if __name__ == '__main__':
    import sys
    sheet(sys.argv[1] if len(sys.argv) > 1 else 'build/ib_sheet.png')
