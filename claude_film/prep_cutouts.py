"""Cut each pose panel out of the reference sheets into an RGBA layer."""
import numpy as np, cv2
from PIL import Image

PANELS = {
    'think': ('assets/poses_a.png', (0, 0, 516, 1024)),
    'point': ('assets/poses_a.png', (517, 0, 1052, 1024)),
    'calm':  ('assets/poses_a.png', (1062, 0, 1536, 1024)),
    'offer': ('assets/poses_b.png', (0, 0, 860, 1024)),
    'heart': ('assets/poses_b.png', (860, 0, 1536, 1024)),
    'full':  ('assets/fullbody.png', (0, 0, 1024, 1536)),
}

def cutout(rgb):
    h, w = rgb.shape[:2]
    lum_min = rgb.min(axis=2)
    near_white = (lum_min >= 236).astype(np.uint8)
    # flood fill background from every border pixel that is near-white
    ff = np.zeros((h + 2, w + 2), np.uint8)
    bg = np.zeros((h, w), np.uint8)
    seeds = [(x, 0) for x in range(0, w, 4)] + [(0, y) for y in range(0, h, 4)] + \
            [(w - 1, y) for y in range(0, h, 4)]
    work = near_white.copy()
    for sx, sy in seeds:
        if work[sy, sx] == 1 and bg[sy, sx] == 0:
            m = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(work, m, (sx, sy), 2, flags=4)
            bg |= (work == 2).astype(np.uint8)
            work[work == 2] = 3
    # soft alpha: background -> 0, with a 1px feathered edge based on whiteness
    alpha = 1.0 - bg.astype(np.float32)
    soft = np.clip((255.0 - lum_min) / 40.0, 0, 1)
    edge = cv2.dilate(bg, np.ones((3, 3), np.uint8)) & (1 - cv2.erode(bg, np.ones((3, 3), np.uint8)))
    alpha = np.where(edge > 0, np.maximum(alpha * 0.5, soft) * (1 - bg * 0.6), alpha)
    # remove tiny islands of foreground (stray specks)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((alpha > 0.5).astype(np.uint8), 8)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] > 60
    alpha *= cv2.dilate(keep[lab].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    # enclosed pure-white pockets (gaps between hair strands) are background too
    pocket = (lum_min >= 244).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pocket, 4)
    big = np.zeros(n, bool); big[1:] = stats[1:, cv2.CC_STAT_AREA] > 150
    pm = big[lab].astype(np.uint8)
    pm_soft = cv2.GaussianBlur(cv2.dilate(pm, np.ones((3, 3), np.uint8)).astype(np.float32), (3, 3), 0)
    alpha = alpha * (1 - np.where(pm_soft > 0, np.maximum(pm.astype(np.float32), pm_soft * (1 - soft)), 0))
    return np.clip(alpha, 0, 1)

if __name__ == '__main__':
    for name, (src, box) in PANELS.items():
        im = np.asarray(Image.open(src).convert('RGB'))[box[1]:box[3], box[0]:box[2]]
        a = cutout(im.astype(np.int32))
        rgba = np.dstack([im, (a * 255).astype(np.uint8)])
        Image.fromarray(rgba, 'RGBA').save(f'build/{name}.png')
        print(name, rgba.shape)
