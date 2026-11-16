"""Claude for the library, cut from the producer's expression sheet ('优雅沉思'), drawn flat to match the room.

Builds build/think2.png (open eyes) and build/think2_closed.png (eyes closed, for blinks and reading), 2x upscaled,
background removed by flood fill from the panel border (so the ivory blouse keeps its body), the hair feathered
where the panel border cuts it, then flattened with one shared palette.
"""
import os
import numpy as np, cv2
from PIL import Image

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'expressions_v2.png')
BOX = (6, 46, 301, 506)          # the 优雅沉思 panel, inside its frame (full-sheet coords)
UP = 2                           # upscale before flattening
# landmarks in full-sheet coords
EYE_R = dict(c=(170.5, 147.5), inner=(155, 153), outer=(187, 142), rx=16, ry=10)
EYE_L = dict(c=(121.5, 166.0), inner=(135, 171), outer=(107, 163), rx=15, ry=9.5)
def to_local(p): return ((p[0] - BOX[0]) * UP, (p[1] - BOX[1]) * UP)

def cutout():
    im = np.asarray(Image.open(SRC).convert('RGB'))[BOX[1]:BOX[3], BOX[0]:BOX[2]].copy()
    im = cv2.resize(im, (im.shape[1] * UP, im.shape[0] * UP), interpolation=cv2.INTER_LANCZOS4)
    h, w = im.shape[:2]
    bg = np.zeros((h + 2, w + 2), np.uint8)
    flood = im.copy()
    seeds = [(x, 0) for x in range(0, w, 7)] + [(0, y) for y in range(0, h, 7)] + [(w - 1, y) for y in range(0, h, 7)]
    for (x, y) in seeds:
        if bg[y + 1, x + 1] == 0 and im[y, x].min() > 225:
            cv2.floodFill(flood, bg, (x, y), (0, 0, 0), (10, 10, 10), (10, 10, 10), cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 8)
    a = 1.0 - bg[1:-1, 1:-1].astype(np.float32) / 255
    # white pockets trapped between strands of hair: near-white islands whose surroundings are mostly hair
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    white = ((im.min(2) > 228) & (im.max(2).astype(int) - im.min(2) < 22)).astype(np.uint8)
    hair = ((hsv[..., 0] < 20) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 90)
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, 8)
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((7, 7), np.uint8)) - comp
        if ring.sum() and (hair | (hsv[..., 2] < 120))[ring > 0].mean() > 0.5 and comp.sum() < 0.02 * comp.size: a[comp > 0] = 0
    # defringe: pull the edge in a little and give edge pixels the colour of the hair just inside
    a = cv2.erode(a, np.ones((3, 3), np.uint8))
    a = cv2.GaussianBlur(a, (0, 0), 0.9)
    inner = (cv2.erode((a > 0.98).astype(np.uint8), np.ones((5, 5), np.uint8))).astype(np.float32)
    num = cv2.GaussianBlur(im.astype(np.float32) * inner[..., None], (0, 0), 4)
    den = cv2.GaussianBlur(inner, (0, 0), 4)[..., None] + 1e-4
    fill = num / den
    wedge = np.clip((0.98 - a) / 0.6, 0, 1)[..., None] * (den > 0.05)
    im = (im * (1 - wedge) + fill * wedge).astype(np.uint8)
    # where the panel border cut her hair: let it fall away softly instead of ending in a straight line
    xx = np.arange(w, dtype=np.float32)[None, :]; yy = np.arange(h, dtype=np.float32)[:, None]
    edge = np.minimum(np.clip(xx / (28 * UP), 0, 1), np.clip((w - 1 - xx) / (28 * UP), 0, 1)) ** 1.5
    edge = np.maximum(edge, np.clip(1 - np.abs(xx - w * 0.52) / (w * 0.30), 0, 1))       # never fade the body itself
    top = np.clip(yy / (6 * UP), 0, 1)
    a = a * edge * top
    return im, a

def close_eyes(im):
    """Eyes closed: the eye painted over with skin (inpaint), then a soft downward lash line."""
    out = im.copy()
    mask = np.zeros(im.shape[:2], np.uint8)
    for E in (EYE_R, EYE_L):
        cx, cy = to_local(E['c'])
        ang = np.degrees(np.arctan2(E['outer'][1] - E['inner'][1], E['outer'][0] - E['inner'][0]))
        cv2.ellipse(mask, (int(cx), int(cy)), (int((E['rx'] + 3) * UP), int((E['ry'] + 3) * UP)), ang if ang > -90 and ang < 90 else ang - 180, 0, 360, 255, -1)
    out = cv2.inpaint(out, mask, 6, cv2.INPAINT_TELEA)
    # soften what inpainting left behind toward a skin tone sampled on the cheek
    for E in (EYE_R, EYE_L):
        p0, p1 = np.array(to_local(E['inner']), np.float64), np.array(to_local(E['outer']), np.float64)
        mid = (p0 + p1) / 2; d = p1 - p0; n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        if n[1] < 0: n = -n                                       # 'down' in the face
        pts = [p0 + (p1 - p0) * u + n * (4.5 * UP) * np.sin(np.pi * u) for u in np.linspace(0, 1, 24)]
        pts = np.array(pts)
        for k_, (wid, col) in enumerate(((3.2 * UP, (66, 30, 24)),)):
            cv2.polylines(out, [np.int32(pts * 4)], False, col, int(wid), cv2.LINE_AA, shift=2)
        # a few lashes at the outer end
        o = pts[-1]
        for j, s in enumerate((0.0, 0.5, 1.0)):
            q = pts[-4 - j * 3]
            tip = q + n * (3.5 * UP) + (p1 - p0) / np.linalg.norm(p1 - p0) * (2.5 * UP * s)
            cv2.line(out, tuple(np.int32(q * 4)), tuple(np.int32(tip * 4)), (66, 30, 24), max(1, int(1.2 * UP)), cv2.LINE_AA, shift=2)
    return out

def flatten(rgbs, a):
    outs, cen = [], None
    for rgb in rgbs:
        ep = cv2.edgePreservingFilter(np.ascontiguousarray(rgb), flags=1, sigma_s=30, sigma_r=0.25)
        Z = ep.reshape(-1, 3).astype(np.float32)
        if cen is None:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, _, cen = cv2.kmeans(Z[a.reshape(-1) > 0.5][::3], 16, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        d = ((Z[:, None, :] - cen[None]) ** 2).sum(-1); q = cen[np.argmin(d, 1)].reshape(ep.shape)
        lum = rgb.astype(np.float32).mean(axis=2); ink = np.clip((95 - lum) / 60, 0, 1)[..., None]
        q = q * (1 - ink * 0.75) + np.array([70, 34, 26]) * ink * 0.75
        outs.append(np.clip(q, 0, 255).astype(np.uint8))
    return outs

def build(force=False):
    here = os.path.dirname(os.path.abspath(__file__))
    p_open, p_closed = os.path.join(here, 'build/think2.png'), os.path.join(here, 'build/think2_closed.png')
    if not force and os.path.exists(p_open) and os.path.exists(p_closed): return
    im, a = cutout()
    o, c = flatten([im, close_eyes(im)], a)
    al = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(np.dstack([o, al])).save(p_open)
    Image.fromarray(np.dstack([c, al])).save(p_closed)

# rig landmarks (2x local coords) so the puppet can breathe, sway its hair and move its head
def rig_pose():
    mid = to_local(((EYE_R['c'][0] + EYE_L['c'][0]) / 2, (EYE_R['c'][1] + EYE_L['c'][1]) / 2))
    return dict(mid=mid, io=52 * UP, mouth=to_local((150, 190)), mw=8 * UP, closed=True, head_r=80 * UP, neck=to_local((166, 228)))

if __name__ == '__main__':
    import sys
    build(force=True)
    S = sys.argv[1]
    for n in ('think2', 'think2_closed'):
        im = np.asarray(Image.open(f'build/{n}.png')).astype(np.float32) / 255
        bg = np.full(im.shape[:2] + (3,), (0.10, 0.08, 0.07), np.float32)
        Image.fromarray((np.clip(im[..., :3] * im[..., 3:] + bg * (1 - im[..., 3:]), 0, 1) * 255).astype(np.uint8)).save(f'{S}/{n}.png')
