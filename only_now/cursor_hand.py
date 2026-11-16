"""The cursor hand, taken straight from the producer's reference (assets/cursor_hand_ref.png, 'CURSOR INDEX / 正常选择').
The outline is traced by hand (its soft render and halo defeat automatic matting); inside it every pixel is the original."""
import numpy as np, cv2
from PIL import Image

OUTLINE = [(98, 118.6), (110, 121), (125.5, 126.8), (138, 127.4), (141.8, 125.5), (146, 128.5), (151, 129.5), (155, 125.5), (158, 123.5), (168, 125.5),
           (178, 128.6), (188, 134.9), (198, 141), (200.5, 149), (203, 160), (205.5, 173), (200.5, 173), (195.5, 175.5),
           (186, 174), (174.3, 170.5), (166, 176), (158, 183), (153, 188), (148, 186.8), (143, 179.3), (135.5, 166.8),
           (138, 160.5), (129.3, 145.5), (121.8, 138), (113, 131.8), (103, 124.3)]
TIP = (98.0, 118.6)
BOX = (94, 114, 210, 192)
UP = 4

_cache = {}
def sprite():
    """Premultiplied RGBA at UP x the reference, the fingertip's position in it, and the silhouette mask."""
    if _cache: return _cache['spr'], _cache['tip'], _cache['mask']
    im = np.asarray(Image.open('assets/cursor_hand_ref.png').convert('RGB')).astype(np.float32) / 255
    x0, y0, x1, y1 = BOX
    c = cv2.resize(im[y0:y1, x0:x1], ((x1 - x0) * UP, (y1 - y0) * UP), interpolation=cv2.INTER_LANCZOS4)
    m = np.zeros(c.shape[:2], np.float32)
    cv2.fillPoly(m, [np.int32((np.array(OUTLINE) - (x0, y0)) * UP * 4)], 1.0, cv2.LINE_AA, shift=2)
    # along the outline, fade out what is only the soft grey halo of the original render
    L = c.mean(2)
    inside = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    edge = 1 - np.clip(inside / (4.5 * UP), 0, 1)
    halo = np.clip((L - 0.45) / 0.15, 0, 1) * np.clip((0.965 - L) / 0.035, 0, 1)
    m = m * (1 - edge * halo)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)
    spr = np.concatenate([np.clip(c, 0, 1) * m[..., None], m[..., None]], 2).astype(np.float32)
    tip = ((TIP[0] - x0) * UP, (TIP[1] - y0) * UP)
    _cache.update(spr=spr, tip=tip, mask=m)
    return spr, tip, m

if __name__ == '__main__':
    import sys
    spr, tip, m = sprite()
    bg = np.full(spr.shape[:2] + (3,), (0.14, 0.11, 0.09), np.float32)
    out = spr[..., :3] + bg * (1 - spr[..., 3:])
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(sys.argv[1])
