"""比特王 BIT KING 開場影片產生器（4 秒、1920x1080、60fps，畫面與音效全部用程式合成）

用法：python brand/make_intro.py [輸出檔.mp4]
需要：numpy、opencv-python-headless、scipy、ffmpeg
流程：宇宙背景（星雲＋星系＋星空）→ 光速躍遷抵達 → 零件像變形金剛一樣飛入、轉動、卡榫
      → 鑽石與文字落位 → 大爆炸（衝擊波、閃光、光芒）→ 掃光收尾
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bitking_intro.mp4")
W, H, FPS, DUR = 1920, 1080, 60, 4.0
NF = int(FPS * DUR)
CX, CY = W / 2, H / 2
TB = 2.56  # 大爆炸（最後卡榫）時間點
rng = np.random.default_rng(7)


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ease_out_cubic(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_quad(x):
    x = clamp01(x)
    return x * x


def ease_out_back(x, s=2.2):
    x = clamp01(x) - 1
    return x * x * ((s + 1) * x + s) + 1


# ───────────────────────── Logo 拆零件 ─────────────────────────
HI = 680  # 零件貼圖解析度（最終 logo 約 640px）
K = HI / 300
G0 = 640 / HI


def load_logo():
    src = cv2.imread(os.path.join(HERE, "logo.png"), cv2.IMREAD_UNCHANGED)
    src = cv2.resize(src, (300, 300), interpolation=cv2.INTER_AREA) if src.shape[0] != 300 else src
    rgba = src[..., [2, 1, 0, 3]].astype(np.float32) / 255
    lum = rgba[..., :3].mean(-1)
    # 中央方塊內的亮色元件＝鑽石與文字
    m = np.zeros((300, 300), np.uint8)
    m[64:237, 60:235] = ((lum > 0.22) & (rgba[..., 3] > 0.5))[64:237, 60:235]
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    gm = np.zeros((300, 300), np.uint8)
    # 群組：1 鑽石頂 2 左 3 中 4 右 5 比 6 特 7 王 8 BIT KING
    for i in range(1, n):
        cx, cy = cen[i]
        if not (90 < cx < 200) or st[i, 4] < 8:
            continue
        if cy < 150:
            g = 1 if (cy < 90 and st[i, 4] > 500) else (2 if cx < 125 else (4 if cx > 168 else 3))
        elif cy < 212:
            g = 5 if cx < 123 else (6 if cx < 170 else 7)
        else:
            g = 8
        gm[lab == i] = g
    gm = cv2.dilate(gm, np.ones((3, 3), np.uint8))
    dark = (lum < 0.25) & (rgba[..., 3] > 0.9)
    dark[:64] = dark[237:] = False
    fill = np.median(rgba[dark][:, :3], axis=0)
    frame = rgba.copy()
    frame[gm > 0, :3] = fill

    def up(layer):
        pm = layer.copy()
        pm[..., :3] *= pm[..., 3:4]
        u = cv2.resize(pm, (HI, HI), interpolation=cv2.INTER_CUBIC)
        u = u + 0.7 * (u - cv2.GaussianBlur(u, (0, 0), 1.6))
        u = np.clip(u, 0, 1)
        u[..., :3] = np.minimum(u[..., :3], u[..., 3:4])
        return u

    full = up(rgba)
    frame_hi = up(frame)
    groups = {}
    for g in range(1, 9):
        layer = rgba.copy()
        layer[..., 3] *= gm == g
        groups[g] = up(layer)
    return full, frame_hi, groups


def make_sprite(img, mask=None):
    spr = img * mask[..., None] if mask is not None else img
    a = spr[..., 3]
    ys, xs = np.where(a > 0.003)
    y0, y1, x0, x1 = max(ys.min() - 2, 0), min(ys.max() + 3, HI), max(xs.min() - 2, 0), min(xs.max() + 3, HI)
    w = a[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    c = np.array([(xx * w).sum() / w.sum(), (yy * w).sum() / w.sum()])
    return dict(spr=np.ascontiguousarray(spr[y0:y1, x0:x1]), o=np.array([x0, y0], float), c=c, mask=mask)


FULL, FRAME_HI, GROUPS = load_logo()
LC = np.array([HI / 2, HI / 2])

# 外框依極座標切成：內圈 6 片、外圈 12 片
yy, xx = np.mgrid[:HI, :HI]
rr = np.hypot(xx - LC[0], yy - LC[1])
ang = (np.degrees(np.arctan2(yy - LC[1], xx - LC[0])) + 360) % 360
R_CORE = 62 * K
core_sec = (((ang + 15) % 360) // 60).astype(int)
outer_sec = (ang // 30).astype(int)
pid = np.where(rr < R_CORE, core_sec, 6 + outer_sec)

pieces = []
events = []  # 給音效用：(時間, 種類, 左右聲道 -1..1)


def add_fly(sp, L, kind, start_dir=None):
    d = sp["c"] - LC
    nd = np.linalg.norm(d)
    d = d / nd if nd > 1e-3 else np.array([0.0, -1.0])
    if start_dir is not None:
        d = np.array(start_dir, float)
        d /= np.linalg.norm(d)
    perp = np.array([-d[1], d[0]])
    far = rng.random() < 0.65
    sp.update(
        L=L, kind=kind, d=d,
        start=d * rng.uniform(1100, 1500) + perp * rng.uniform(-450, 450),
        stage=d * (60 if kind != "diamond" else 45),
        s0=rng.uniform(0.35, 0.7) if far else rng.uniform(1.7, 2.3),
        a0=rng.choice([-1, 1]) * rng.uniform(1.2, 2.4) * math.pi,
        a_st=rng.choice([-1, 1]) * rng.uniform(0.35, 0.6),
        phi0=rng.uniform(1.5, 3.0) * math.pi,
        glint=rng.uniform(0, 2 * math.pi),
    )
    pieces.append(sp)


T_FLY, T_SNAP, T_SLIDE = 0.5, 0.14, 0.07
for i, L in zip([0, 3, 1, 4, 2, 5], [0.80, 0.80, 0.92, 0.92, 1.04, 1.04]):
    add_fly(make_sprite(FRAME_HI, (pid == i).astype(np.float32)), L, "core")
order = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
for j, k in enumerate(order):
    add_fly(make_sprite(FRAME_HI, (pid == 6 + k).astype(np.float32)), 1.20 + (j // 2) * 0.115, "outer")
for g, L, sd in [(1, 1.96, (0.1, -1)), (2, 2.04, (-0.6, -1)), (4, 2.04, (0.6, -1)), (3, 2.12, (0, -1))]:
    add_fly(make_sprite(GROUPS[g]), L, "diamond", sd)
for g, L in [(5, 2.24), (6, 2.30), (7, 2.36), (8, 2.45)]:
    sp = make_sprite(GROUPS[g])
    sp.update(L=L, kind="text")
    pieces.append(sp)

for p in pieces:
    px = float(np.clip((p["c"][0] - LC[0]) / 300, -1, 1))
    if p["kind"] == "text":
        events += [(p["L"] - 0.12, "stamp_whoosh", px), (p["L"], "stamp", px)]
    else:
        t0 = p["L"] - T_SLIDE - T_SNAP - T_FLY
        events += [(t0, "whoosh", float(np.clip(p["d"][0] + p["start"][0] / 2000, -1, 1))),
                   (p["L"] - T_SLIDE - T_SNAP, "servo", px), (p["L"], "clank", px)]


def piece_state(p, t):
    """回傳 (位移, 旋轉, 縮放, 翻轉 sx, 透明度, 動態模糊?)；None＝還沒出現；'locked'＝已卡榫"""
    L = p["L"]
    if t >= L:
        return "locked"
    if p["kind"] == "text":
        u = (t - (L - 0.12)) / 0.12
        if u < 0:
            return None
        e = ease_in_quad(u)
        return np.zeros(2), 0.0, 2.8 - 1.8 * e, 1.0, clamp01(u * 2.5), False
    t0 = L - T_SLIDE - T_SNAP - T_FLY
    if t < t0:
        return None
    if t < t0 + T_FLY:
        e = ease_out_cubic((t - t0) / T_FLY)
        pos = p["start"] + (p["stage"] - p["start"]) * e
        rot = p["a0"] + (p["a_st"] - p["a0"]) * e
        return pos, rot, p["s0"] + (1 - p["s0"]) * e, math.cos(p["phi0"] * (1 - e)), 1.0, True
    if t < L - T_SLIDE:
        # 變形金剛式「喀、喀」兩段式轉動
        u = (t - (t0 + T_FLY)) / T_SNAP
        if u < 0.5:
            rot = p["a_st"] * (1 - 0.5 * ease_out_back(u / 0.5))
        else:
            rot = p["a_st"] * 0.5 * (1 - ease_out_back((u - 0.5) / 0.5))
        return p["stage"].copy(), rot, 1.0, 1.0, 1.0, False
    u = (t - (L - T_SLIDE)) / T_SLIDE
    return p["stage"] * (1 - ease_in_quad(u)), 0.0, 1.0, 1.0, 1.0, False


# ───────────────────────── 宇宙背景 ─────────────────────────
BW, BH = 2304, 1296


def fbm(h, w, octaves, base, seed, gain=0.55):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gw = int(base * 2 ** o) + 2
        gh = int(gw * h / w) + 2
        out += amp * cv2.resize(r.random((gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= gain
    return out / tot


def splat(img, xs, ys, cols):
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    ok = (xi >= 0) & (xi < img.shape[1]) & (yi >= 0) & (yi < img.shape[0])
    for ch in range(3):
        np.add.at(img[..., ch], (yi[ok], xi[ok]), cols[ok, ch])


def star_colors(r, n):
    t = r.random(n)
    warm = np.array([1.0, 0.82, 0.6])
    cool = np.array([0.7, 0.82, 1.0])
    return (cool[None] * (1 - t[:, None]) + warm[None] * t[:, None]) ** 0.7


def blob(img, cx, cy, sx, sy, rot, col, amp):
    rx = int(max(sx, sy) * 3.5) + 2
    x0, x1, y0, y1 = int(cx - rx), int(cx + rx), int(cy - rx), int(cy + rx)
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    gx, gy = gx - cx, gy - cy
    c, s = math.cos(rot), math.sin(rot)
    u, v = c * gx + s * gy, -s * gx + c * gy
    g = amp * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    ys0, xs0 = max(y0, 0), max(x0, 0)
    ys1, xs1 = min(y1, img.shape[0]), min(x1, img.shape[1])
    img[ys0:ys1, xs0:xs1] += g[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0, None] * np.array(col, np.float32)


def galaxy(img, cx, cy, R, tilt, rot, n, seed, arms=2, wind=1.7):
    r = np.random.default_rng(seed)
    rad = np.minimum(r.exponential(0.32, n), 1.3) * R
    arm = r.integers(0, arms, n)
    theta = arm * 2 * math.pi / arms + wind * np.log1p(rad / (0.06 * R)) + r.normal(0, 0.28, n)
    disk = r.random(n) < 0.3
    theta[disk] = r.random(disk.sum()) * 2 * math.pi
    x, y = rad * np.cos(theta), rad * np.sin(theta) * tilt
    c, s = math.cos(rot), math.sin(rot)
    X, Y = cx + c * x - s * y, cy + s * x + c * y
    k = np.clip(rad / R, 0, 1)[:, None]
    cols = np.array([1.0, 0.85, 0.62])[None] * (1 - k) + np.array([0.55, 0.72, 1.0])[None] * k
    hii = (r.random(n) < 0.04) & ~disk & (rad > 0.25 * R)
    cols[hii] = [1.0, 0.35, 0.6]
    cols *= (0.05 + 0.12 * r.random(n))[:, None]
    layer = np.zeros_like(img)
    splat(layer, X, Y, cols)
    layer = cv2.GaussianBlur(layer, (0, 0), 1.0) + 0.6 * cv2.GaussianBlur(layer, (0, 0), 4.0)
    blob(layer, cx, cy, 0.10 * R, 0.10 * R * tilt, rot, (1.0, 0.86, 0.65), 1.1)
    blob(layer, cx, cy, 0.035 * R, 0.035 * R * tilt, rot, (1.0, 0.97, 0.9), 1.5)
    blob(layer, cx, cy, 0.45 * R, 0.45 * R * tilt, rot, (0.45, 0.5, 0.75), 0.10)
    img += layer


def build_background():
    print("建立宇宙背景…", flush=True)
    r = np.random.default_rng(11)
    yy, xx = np.mgrid[:BH, :BW].astype(np.float32)
    # 銀河帶：左上到右下
    p0, p1 = np.array([0, 0.12 * BH]), np.array([BW, 0.88 * BH])
    dvec = (p1 - p0) / np.linalg.norm(p1 - p0)
    dist = np.abs((xx - p0[0]) * dvec[1] - (yy - p0[1]) * dvec[0])
    warp = (fbm(BH, BW, 4, 3, 5) - 0.5) * 420
    band = np.exp(-((dist + warp) / 330) ** 2)
    n1, n2, n3 = fbm(BH, BW, 7, 4, 1), fbm(BH, BW, 7, 5, 2), fbm(BH, BW, 6, 6, 3)
    neb1 = np.clip((n1 - 0.38) * 2.6, 0, 1) ** 2 * band
    neb2 = np.clip((n2 - 0.45) * 3.0, 0, 1) ** 2 * (0.35 + band)
    dust = 1 - 0.75 * np.clip((n3 - 0.5) * 4, 0, 1) * band
    far = (neb1[..., None] * np.array([0.10, 0.33, 0.55]) + neb2[..., None] * np.array([0.38, 0.08, 0.42])
           + (neb1 * neb2)[..., None] * np.array([0.9, 0.45, 0.3]) * 1.5)
    far = far * dust[..., None] + band[..., None] * np.array([0.03, 0.035, 0.06])
    far = far.astype(np.float32)
    # 遠方星空（銀河帶附近更密）
    ns = 9000
    xs, ys = r.random(ns) * BW, r.random(ns) * BH
    t = r.random(3500) * 1.3 - 0.15
    off = r.normal(0, 200, 3500)
    bx, by = p0[0] + (p1 - p0)[0] * t - dvec[1] * off, p0[1] + (p1 - p0)[1] * t + dvec[0] * off
    xs, ys = np.concatenate([xs, bx]), np.concatenate([ys, by])
    b = np.minimum(0.08 + 0.5 * r.pareto(2.6, len(xs)), 1.4)
    stars = np.zeros_like(far)
    splat(stars, xs, ys, star_colors(r, len(xs)) * b[:, None])
    far += cv2.GaussianBlur(stars, (0, 0), 0.6) * 2.2
    # 星系
    gal = np.zeros_like(far)
    galaxy(gal, 0.80 * BW, 0.24 * BH, 250, 0.42, math.radians(-28), 150000, 21)
    galaxy(gal, 0.14 * BW, 0.79 * BH, 95, 0.30, math.radians(35), 40000, 22, wind=2.2)
    galaxy(gal, 0.40 * BW, 0.90 * BH, 38, 0.6, math.radians(80), 8000, 23)
    for _ in range(14):  # 遠處小星系
        blob(gal, r.random() * BW, r.random() * BH, r.uniform(3, 7), r.uniform(1, 3), r.random() * 3,
             (0.8, 0.8, 1.0), r.uniform(0.15, 0.35))
    far += gal
    # 近景亮星（視差較大）＋繞射星芒
    near = np.zeros_like(far)
    for _ in range(260):
        x, y = r.random() * BW, r.random() * BH
        col = star_colors(r, 1)[0]
        blob(near, x, y, r.uniform(0.7, 1.3), r.uniform(0.7, 1.3), 0, col, r.uniform(0.4, 1.0))
    for _ in range(18):
        x, y = r.random() * BW, r.random() * BH
        col = star_colors(r, 1)[0]
        a = r.uniform(0.8, 1.6)
        blob(near, x, y, 2.0, 2.0, 0, col, a)
        blob(near, x, y, 7, 7, 0, col, 0.15 * a)
        ln = r.uniform(25, 70)
        blob(near, x, y, ln, 0.8, 0, col, 0.5 * a)
        blob(near, x, y, ln, 0.8, math.pi / 2, col, 0.5 * a)
    return far, near


FAR, NEAR = build_background()

# 預先計算：暈影、極座標距離、橫向光暈
gy, gx = np.mgrid[:H, :W].astype(np.float32)
VIG = (1 - 0.55 * smooth(0.45, 1.25, np.hypot((gx - CX) / CX, (gy - CY) / CY)))[..., None].astype(np.float32)

# 光速躍遷星點
NW = 700
WX, WY = rng.uniform(-1, 1, NW) * 2.2, rng.uniform(-1, 1, NW) * 1.3
WZ = rng.uniform(0.3, 7.0, NW)
WB = rng.uniform(0.3, 1.0, NW)


def warp_dist(t):
    # 速度由快到 0（0.9 秒停下），回傳累計前進距離
    T = 0.9
    t = min(t, T)
    return 6.0 * (1 - (1 - t / T) ** 3)


# 零件的「全組裝」貼圖（避免卡榫後出現接縫）
locked_cache = {}


def locked_sprite(locked_ids):
    key = tuple(locked_ids)
    if key not in locked_cache:
        img = np.zeros((HI, HI, 4), np.float32)
        frame_mask = np.zeros((HI, HI), np.float32)
        for i in locked_ids:
            p = pieces[i]
            if p["mask"] is not None:
                frame_mask += p["mask"]
        img += FRAME_HI * frame_mask[..., None]
        for i in locked_ids:
            p = pieces[i]
            if p["mask"] is None:
                s = p["spr"]
                o = p["o"].astype(int)
                roi = img[o[1]:o[1] + s.shape[0], o[0]:o[0] + s.shape[1]]
                roi[:] = s + roi * (1 - s[..., 3:4])
        locked_cache.clear()
        locked_cache[key] = img
    return locked_cache[key]


def blit(dst, spr, A, P, mode="over", rgb_gain=1.0, a_gain=1.0):
    h, w = spr.shape[:2]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32) @ A.T + P
    x0, x1 = max(int(corners[:, 0].min()) - 2, 0), min(int(corners[:, 0].max()) + 3, dst.shape[1])
    y0, y1 = max(int(corners[:, 1].min()) - 2, 0), min(int(corners[:, 1].max()) + 3, dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    M = np.hstack([A, (P - np.array([x0, y0]))[:, None]]).astype(np.float32)
    out = cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    roi = dst[y0:y1, x0:x1]
    rgb = out[..., :3] * np.asarray(rgb_gain, np.float32) * a_gain
    if mode == "over":
        roi[..., :3] = rgb + roi[..., :3] * (1 - out[..., 3:4] * a_gain)
    elif mode == "acc":
        roi[..., :3] += rgb
        roi[..., 3] += out[..., 3] * a_gain
    else:
        roi[..., :3] += rgb


def piece_xform(p, G, C, st):
    pos, rot, s, sx, _, _ = st
    c, sn = math.cos(rot), math.sin(rot)
    A = np.array([[c, -sn], [sn, c]]) @ np.diag([sx * s * G, s * G])
    P = C + G * (p["c"] - LC) + pos * (G / G0) + A @ (p["o"] - p["c"])
    return A, P


# 火花
sparks = {k: np.zeros((0, 2), np.float32) for k in ("p", "v")}
sparks["age"] = np.zeros(0, np.float32)
sparks["life"] = np.zeros(0, np.float32)
sparks["col"] = np.zeros((0, 3), np.float32)


def spawn_sparks(pt, n, big=False):
    ang = rng.uniform(0, 2 * math.pi, n)
    spd = rng.uniform(250, 1300 if big else 900, n)
    v = np.stack([np.cos(ang), np.sin(ang)], 1) * spd[:, None]
    p = pt + rng.normal(0, 18, (n, 2))
    col = np.where(rng.random(n)[:, None] < 0.7, [1.0, 0.75, 0.4], [0.6, 0.85, 1.0])
    sparks["p"] = np.concatenate([sparks["p"], p]).astype(np.float32)
    sparks["v"] = np.concatenate([sparks["v"], v]).astype(np.float32)
    sparks["age"] = np.concatenate([sparks["age"], np.zeros(n)]).astype(np.float32)
    sparks["life"] = np.concatenate([sparks["life"], rng.uniform(0.25, 0.6 if big else 0.4, n)]).astype(np.float32)
    sparks["col"] = np.concatenate([sparks["col"], col]).astype(np.float32)


# 掃光用座標
SWEEP_U = ((xx + 0.55 * yy) / (HI * 1.55)).astype(np.float32)
GLOW_SPR = np.zeros((HI, HI, 4), np.float32)
GLOW_SPR[..., 3] = cv2.GaussianBlur(FULL[..., 3], (0, 0), 28)
GLOW_SPR[..., :3] = GLOW_SPR[..., 3:4] * np.array([0.25, 0.6, 1.0], np.float32)


def render_frame(f):
    t = f / FPS
    dt = 1 / FPS
    tb = t - TB
    # 鏡頭：慢慢推進＋爆炸時震動
    shake = np.zeros(2)
    if tb > 0:
        amp = 22 * math.exp(-tb / 0.16)
        shake = np.array([math.sin(t * 91.0) + 0.5 * math.sin(t * 173.0), math.cos(t * 83.0) + 0.5 * math.sin(t * 157.0)]) * amp
    for p in pieces:  # 卡榫瞬間的小震動
        dl = t - p["L"]
        if 0 <= dl < 0.08 and p["kind"] != "text":
            shake += rng.normal(0, 3.0, 2) * (1 - dl / 0.08)
    C = np.array([CX, CY]) + shake
    push = t / DUR
    # 背景
    frame = np.zeros((H, W, 3), np.float32)
    for layer, z in ((FAR, 0.9 * (1 + 0.05 * push)), (NEAR, 0.9 * (1 + 0.16 * push))):
        punch = 1 + (0.03 * math.exp(-tb / 0.2) if tb > 0 else 0)
        s = z * punch
        M = np.array([[s, 0, C[0] - s * BW / 2], [0, s, C[1] - s * BH / 2]], np.float32)
        frame += cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR)
    frame *= 0.85 + 0.25 * (math.exp(-tb / 0.8) if tb > 0 else 0)

    glow = np.zeros((H, W, 3), np.float32)
    # 光速躍遷
    if t < 1.05:
        d1, d0 = warp_dist(t), warp_dist(max(t - 4 * dt, 0))
        z1, z0 = WZ - d1, WZ - d0
        fade = 1 - smooth(0.75, 1.05, t)
        lines = np.zeros((H, W, 3), np.uint8)
        ok = (z1 > 0.05) & (z0 > 0.05)
        for i in np.where(ok)[0]:
            x1, y1 = C[0] + 700 * WX[i] / z1[i], C[1] + 700 * WY[i] / z1[i]
            x0, y0 = C[0] + 700 * WX[i] / z0[i], C[1] + 700 * WY[i] / z0[i]
            if not (-200 < x1 < W + 200 and -200 < y1 < H + 200):
                continue
            v = int(255 * WB[i] * min(1, 1.2 / z1[i]))
            cv2.line(lines, (int(x0 * 4), int(y0 * 4)), (int(x1 * 4), int(y1 * 4)), (int(v * 0.8), int(v * 0.88), v),
                     2 if z1[i] < 1.2 else 1, cv2.LINE_AA, shift=2)
        glow += lines.astype(np.float32) / 255 * 1.4 * fade
        # 中央隧道光
        tun = math.exp(-t / 0.25) * 0.8
        frame += tun * np.exp(-((gx - C[0]) ** 2 + (gy - C[1]) ** 2) / (2 * 260 ** 2))[..., None] * np.array([0.35, 0.55, 1.0], np.float32)

    G = G0 * (1 + 0.04 * smooth(TB, DUR, t)) * (1 + (0.07 * math.exp(-tb / 0.12) if tb > 0 else 0))

    # HUD 光環（組裝期間）
    hud_a = smooth(0.25, 0.6, t) * (1 - smooth(TB - 0.08, TB + 0.02, t))
    if hud_a > 0.01:
        hud = np.zeros((H, W, 3), np.uint8)
        sc = 1 - 0.12 * smooth(TB - 0.4, TB, t)
        cc = (int(C[0] * 4), int(C[1] * 4))
        col = (70, 190, 255)
        for k in range(3):
            st = (t * 70 + k * 120) % 360
            cv2.ellipse(hud, cc, (int(375 * sc * 4),) * 2, 0, st, st + 70, col, 2, cv2.LINE_AA, shift=2)
        for k in range(60):
            a = math.radians(k * 6 - t * 30)
            r1, r2 = 405 * sc, (418 if k % 5 == 0 else 411) * sc
            p1 = (int((C[0] + r1 * math.cos(a)) * 4), int((C[1] + r1 * math.sin(a)) * 4))
            p2 = (int((C[0] + r2 * math.cos(a)) * 4), int((C[1] + r2 * math.sin(a)) * 4))
            cv2.line(hud, p1, p2, col, 1, cv2.LINE_AA, shift=2)
        cv2.circle(hud, cc, int(345 * sc * 4), (40, 110, 160), 1, cv2.LINE_AA, shift=2)
        glow += hud.astype(np.float32) / 255 * 0.55 * hud_a

    # Logo 背光
    if tb > -0.3:
        gI = (smooth(-0.3, 0, tb) * 0.5 + (1.2 * math.exp(-tb / 0.4) if tb > 0 else 0)) * (1 + 0.08 * math.sin(t * 9))
        A = np.eye(2) * G * 1.08
        blit(frame, GLOW_SPR, A, C - A @ LC, mode="add", a_gain=gI)

    # 已卡榫的零件
    locked = [i for i, p in enumerate(pieces) if t >= p["L"]]
    if locked:
        A = np.eye(2) * G
        P = C - A @ LC
        img = locked_sprite(locked)
        blit(frame, img, A, P, mode="over")
        # 掃光
        if 2.95 < t < 3.6:
            pos = -0.15 + 1.3 * (t - 2.95) / 0.6
            band = np.exp(-((SWEEP_U - pos) / 0.05) ** 2)
            sw = img * band[..., None]
            blit(frame, sw, A, P, mode="add", rgb_gain=np.array([1.2, 1.25, 1.35]))

    # 飛行中的零件
    acc = np.zeros((H, W, 4), np.float32)
    for i, p in enumerate(pieces):
        stt = piece_state(p, t)
        if stt is None:
            continue
        if stt == "locked":
            dl = t - p["L"]
            if dl < dt * 0.999:
                ctr = C + G * (p["c"] - LC) * 0.85
                spawn_sparks(ctr, 26 if p["kind"] != "text" else 40)
            if dl < 0.35 and t < TB + 0.05:
                A, P = piece_xform(p, G, C, (np.zeros(2), 0.0, 1.0, 1.0, 1, False))
                blit(glow, p["spr"], A, P, mode="add", rgb_gain=np.array([0.4, 0.8, 1.0]) * 1.6 * (1 - dl / 0.35) ** 2)
            continue
        subs = [0, 1 / 3, 2 / 3] if stt[5] else [0]
        for sdt in subs:
            st = piece_state(p, t - sdt * dt * 1.5) if sdt else stt
            if st is None or st == "locked":
                continue
            A, P = piece_xform(p, G, C, st)
            gl = 1 + 0.9 * max(0.0, math.cos(2 * st[1] + p.get("glint", 0))) ** 8
            blit(acc, p["spr"], A, P, mode="acc", rgb_gain=gl, a_gain=st[4] / len(subs))
    a = np.clip(acc[..., 3:4], 0, 1)
    frame = acc[..., :3] + frame * (1 - a)

    # 爆炸瞬間的火花
    if abs(t - TB) < dt / 2:
        spawn_sparks(C, 260, big=True)
    if len(sparks["age"]):
        sparks["age"] += dt
        alive = sparks["age"] < sparks["life"]
        for k in ("p", "v", "age", "life", "col"):
            sparks[k] = sparks[k][alive]
        sparks["v"] *= math.exp(-2.5 * dt)
        sparks["v"][:, 1] += 380 * dt
        prev = sparks["p"].copy()
        sparks["p"] += sparks["v"] * dt
        lines = np.zeros((H, W, 3), np.uint8)
        fadeK = (1 - sparks["age"] / sparks["life"]) ** 1.5
        for j in range(len(fadeK)):
            c = (sparks["col"][j] * 255 * fadeK[j]).astype(int)
            p0, p1 = prev[j] - sparks["v"][j] * dt * 1.5, sparks["p"][j]
            cv2.line(lines, (int(p0[0] * 4), int(p0[1] * 4)), (int(p1[0] * 4), int(p1[1] * 4)),
                     tuple(int(x) for x in c), 2, cv2.LINE_AA, shift=2)
        glow += lines.astype(np.float32) / 255 * 1.6

    # 光暈（bloom）
    if glow.any():
        small = cv2.resize(glow, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bl = cv2.GaussianBlur(small, (0, 0), 2.5) * 1.2 + cv2.GaussianBlur(small, (0, 0), 9) * 1.0
        frame += glow + cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR)

    if tb > -0.05:
        # 光芒（god rays）
        rI = (0.9 * math.exp(-max(tb, 0) / 0.35) + 0.22 * smooth(0, 0.3, tb)) * smooth(-0.05, 0.0, tb)
        if rI > 0.01:
            q = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
            src = np.clip(q - 0.55, 0, None)
            src[..., :] = src.mean(-1, keepdims=True)
            acc_r = np.zeros_like(src)
            cq = C / 4
            for k in range(16):
                s = 1 + 0.9 * k / 15
                M = np.array([[s, 0, cq[0] * (1 - s)], [0, s, cq[1] * (1 - s)]], np.float32)
                acc_r += cv2.warpAffine(src, M, (W // 4, H // 4)) * (1 - k / 16)
            acc_r = cv2.GaussianBlur(acc_r, (0, 0), 1.5) / 6
            frame += cv2.resize(acc_r, (W, H)) * rI * np.array([0.75, 0.88, 1.0], np.float32)
    if tb > 0:
        d = np.hypot(gx - C[0], gy - C[1])
        # 衝擊波（扭曲＋光環）
        if tb < 1.1:
            rad = 60 + 2100 * (1 - math.exp(-tb / 0.45))
            wid = 50 + 60 * tb
            ring = np.exp(-((d - rad) / wid) ** 2)
            disp = 38 * ring * math.exp(-tb / 0.5)
            ux, uy = (gx - C[0]) / (d + 1e-3), (gy - C[1]) / (d + 1e-3)
            frame = cv2.remap(frame, gx - ux * disp, gy - uy * disp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            rI = 0.9 * math.exp(-tb / 0.35)
            thin = np.exp(-((d - rad) / (8 + 20 * tb)) ** 2)
            frame += (thin * rI + ring * rI * 0.25)[..., None] * np.array([0.5, 0.8, 1.0], np.float32)
        # 閃光
        fl = 1.6 * math.exp(-tb / 0.07)
        frame += fl * (0.4 + 0.6 * np.exp(-(d / 700) ** 2))[..., None] * np.array([0.85, 0.92, 1.0], np.float32)
        # 橫向鏡頭光暈
        an = 1.1 * math.exp(-tb / 0.5) + 0.12
        streak = np.exp(-((gy - C[1]) / 3.5) ** 2) * np.exp(-np.abs(gx - C[0]) / 650)
        streak += 0.3 * np.exp(-((gy - C[1]) / 18) ** 2) * np.exp(-np.abs(gx - C[0]) / 400)
        frame += (streak * an)[..., None] * np.array([0.35, 0.65, 1.0], np.float32)
        # 色差
        ca = 10 * math.exp(-tb / 0.18)
        if ca > 0.3:
            for ch, sgn in ((0, 1), (2, -1)):
                s = 1 + sgn * ca / 900
                M = np.array([[s, 0, C[0] * (1 - s)], [0, s, C[1] * (1 - s)]], np.float32)
                frame[..., ch] = cv2.warpAffine(np.ascontiguousarray(frame[..., ch]), M, (W, H), borderMode=cv2.BORDER_REFLECT)

    frame *= VIG
    frame *= smooth(0.0, 0.18, t) * (1 - smooth(3.7, 4.0, t))
    # 柔和壓高光
    hi = frame > 0.82
    frame[hi] = 0.82 + 0.18 * np.tanh((frame[hi] - 0.82) / 0.18)
    return (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)


# ───────────────────────── 音效 ─────────────────────────
SR = 48000


def make_audio(path):
    print("合成音效…", flush=True)
    r = np.random.default_rng(3)
    N = int(SR * (DUR + 2.5))  # 多留尾巴給殘響，最後再裁掉
    dry = np.zeros((N, 2))
    send = np.zeros((N, 2))  # 送進殘響
    tt = np.arange(N) / SR

    def place(buf, t0, sig, pan=0.0, gain=1.0):
        i0 = int(t0 * SR)
        if sig.ndim == 1:
            sig = np.stack([sig, sig], 1)
        lg, rg = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        sig = sig * np.array([lg, rg]) * math.sqrt(2) * gain
        j0 = max(i0, 0)
        j1 = min(i0 + len(sig), N)
        if j1 > j0:
            buf[j0:j1] += sig[j0 - i0:j1 - i0]

    def env(n, a, d, shape=1.0):
        t = np.arange(n) / SR
        return np.minimum(t / max(a, 1e-4), 1) * np.exp(-np.maximum(t - a, 0) / d) ** shape

    def sweep_noise(dur, f0, f1, q=1.5, stereo=False):
        n = int(dur * SR)
        out = []
        for ch in range(2 if stereo else 1):
            x = r.standard_normal(n)
            y = np.zeros(n)
            zi = None
            blk = 256
            for b in range(0, n, blk):
                fc = f0 * (f1 / f0) ** (b / n)
                lo, hi = max(fc / q, 20), min(fc * q, SR / 2 - 100)
                sos = signal.butter(2, [lo, hi], "bandpass", fs=SR, output="sos")
                if zi is None:
                    zi = np.zeros((sos.shape[0], 2))
                y[b:b + blk], zi = signal.sosfilt(sos, x[b:b + blk], zi=zi)
            out.append(y / (np.std(y) + 1e-9))
        return np.stack(out, 1) if stereo else out[0]

    def lp(x, fc, order=4):
        return signal.sosfilt(signal.butter(order, fc, "lowpass", fs=SR, output="sos"), x, axis=0)

    def hp(x, fc, order=4):
        return signal.sosfilt(signal.butter(order, fc, "highpass", fs=SR, output="sos"), x, axis=0)

    def glide_sine(dur, f0, f1, curve="exp"):
        n = int(dur * SR)
        t = np.arange(n) / SR
        f = f0 * (f1 / f0) ** (t / dur) if curve == "exp" else f0 + (f1 - f0) * t / dur
        return np.sin(2 * np.pi * np.cumsum(f) / SR)

    # 1) 光速躍遷抵達
    n = int(1.1 * SR)
    wh = sweep_noise(1.1, 7000, 250, 1.4, stereo=True) * (env(n, 0.05, 0.35)[:, None])
    place(dry, 0.0, wh, gain=0.35)
    place(send, 0.0, wh, gain=0.2)
    tone = glide_sine(1.0, 1400, 70) * env(int(1.0 * SR), 0.02, 0.3)
    place(dry, 0.0, tone, gain=0.08)
    sub = glide_sine(1.0, 120, 38) * env(int(1.0 * SR), 0.01, 0.45)
    place(dry, 0.0, np.tanh(sub * 2), gain=0.35)

    # 2) 低頻嗡鳴＋上升張力（爆炸前 30ms 全部切掉，製造落差）
    cut = TB - 0.03
    n = int((cut - 0.2) * SR)
    t = np.arange(n) / SR
    ramp = (t / t[-1]) ** 1.5
    drone = np.stack([np.sin(2 * np.pi * 36 * t) + 0.6 * np.sin(2 * np.pi * 54 * t),
                      np.sin(2 * np.pi * 36.6 * t) + 0.6 * np.sin(2 * np.pi * 54.4 * t)], 1)
    rumble = lp(r.standard_normal((n, 2)), 140)
    rumble /= np.std(rumble)
    fade_edge = np.minimum(1, (n - np.arange(n)) / (0.005 * SR))[:, None]
    place(dry, 0.2, (drone * 0.25 + rumble * 0.12) * (0.3 + 0.7 * ramp)[:, None] * fade_edge)
    # 上升音（鋸齒波和聲）
    rs = 1.35
    n = int((cut - rs) * SR)
    t = np.arange(n) / SR
    f = 90 * (4.0) ** (t / t[-1])
    ph = 2 * np.pi * np.cumsum(f) / SR
    saw = sum(np.sin(k * ph) / k for k in range(1, 9)) + sum(np.sin(k * ph * 1.007) / k for k in range(1, 9))
    saw = lp(saw, 2500) * (t / t[-1]) ** 2.5
    nr = hp(r.standard_normal((n, 2)), 3000) * ((t / t[-1]) ** 3)[:, None]
    fade_edge = np.minimum(1, (n - np.arange(n)) / (0.004 * SR))
    place(dry, rs, saw * fade_edge, gain=0.13)
    place(dry, rs, nr * fade_edge[:, None], gain=0.07)
    place(send, rs, saw * fade_edge, gain=0.05)
    # 反向吸氣
    n = int(0.35 * SR)
    rev = (sweep_noise(0.35, 800, 6000, 2.0, stereo=True) * np.exp(-np.arange(n) / (0.09 * SR))[:, None])[::-1]
    place(dry, cut - 0.35, rev, gain=0.3)

    # 3) 零件：飛行呼嘯、伺服馬達、金屬卡榫
    def clank(f0, decay=1.0, heavy=False):
        n = int(0.9 * SR * decay)
        t = np.arange(n) / SR
        ratios = [1, 2.32, 4.25, 6.63, 9.38, 12.1]
        amps = [1, 0.75, 0.55, 0.4, 0.28, 0.18]
        decs = [0.30, 0.22, 0.16, 0.11, 0.08, 0.06]
        y = sum(a * np.sin(2 * np.pi * f0 * rt * t * (1 + 0.002 * r.standard_normal())) * np.exp(-t / (d * decay))
                for rt, a, d in zip(ratios, amps, decs))
        y += hp(r.standard_normal(n), 2500) * np.exp(-t / 0.006) * 1.5
        y += np.sin(2 * np.pi * (55 if heavy else 75) * t) * np.exp(-t / (0.12 if heavy else 0.07)) * (2.5 if heavy else 1.6)
        return np.tanh(y * 0.8)

    def servo(dur):
        n = int(dur * SR)
        out = np.zeros(n)
        half = n // 2
        for k, (a, b) in enumerate([(320, 760), (420, 980)]):
            m = half
            t = np.arange(m) / SR
            f = a + (b - a) * (t / t[-1]) ** 0.6
            sq = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR))
            sq = lp(sq, 2400) * np.sin(np.pi * t / t[-1]) ** 0.5
            out[k * half:k * half + m] += sq * 0.5
            for c in range(4):  # 棘輪喀喀聲
                i = k * half + int(c * 0.012 * SR)
                ln = int(0.004 * SR)
                if i + ln < n:
                    out[i:i + ln] += hp(r.standard_normal(ln), 3000) * np.exp(-np.arange(ln) / (0.001 * SR)) * 1.2
            i = k * half + m - int(0.006 * SR)
            ln = int(0.03 * SR)
            if i + ln < n:
                out[i:i + ln] += np.tanh(hp(r.standard_normal(ln), 800) * np.exp(-np.arange(ln) / (0.004 * SR)) * 3)
        return out

    for (te, kind, pan) in events:
        if kind == "whoosh":
            n = int(T_FLY * SR)
            e = (np.arange(n) / n) ** 1.5 * np.exp(-np.maximum(np.arange(n) / n - 0.8, 0) * 10)
            w = sweep_noise(T_FLY, 300, 2200, 1.6) * e
            place(dry, te, w, pan, 0.06)
            place(send, te, w, pan, 0.04)
        elif kind == "servo":
            s = servo(T_SNAP)
            place(dry, te, s, pan * 0.7, 0.10)
            place(send, te, s, pan * 0.7, 0.04)
        elif kind == "clank":
            c = clank(r.uniform(170, 300))
            place(dry, te, c, pan * 0.8, 0.30)
            place(send, te, c, pan * 0.8, 0.18)
        elif kind == "stamp_whoosh":
            w = sweep_noise(0.12, 4000, 600, 1.5) * np.linspace(0, 1, int(0.12 * SR)) ** 2
            place(dry, te, w, pan, 0.10)
        elif kind == "stamp":
            c = clank(r.uniform(110, 140), 1.4, heavy=True)
            place(dry, te, c, pan * 0.5, 0.42)
            place(send, te, c, pan * 0.5, 0.25)

    # 4) 大爆炸
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    f = 28 + 85 * np.exp(-t / 0.10)
    boom_sub = np.tanh(2.2 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)) * 1.0
    click = hp(r.standard_normal(n), 1500) * np.exp(-t / 0.004) * 1.5
    crash = lp(r.standard_normal((n, 2)), 3500) * np.exp(-t / 0.35)[:, None] * 0.9
    air = hp(r.standard_normal((n, 2)), 5000) * np.exp(-t / 1.1)[:, None] * 0.25
    ring = sum(a * np.sin(2 * np.pi * 98 * rt * t) * np.exp(-t / d)
               for rt, a, d in zip([1, 2.32, 4.25, 6.63, 1.5], [1, 0.6, 0.4, 0.25, 0.5], [1.6, 1.1, 0.8, 0.5, 1.3]))
    boom = np.stack([boom_sub, boom_sub], 1) + click[:, None] + crash + air + 0.35 * ring[:, None]
    place(dry, TB, boom, gain=0.85)
    place(send, TB, crash + air + 0.4 * ring[:, None], gain=0.45)

    # 5) 收尾：明亮和弦＋掃光「鏘」聲＋星光閃爍
    n = int((DUR + 1.0 - TB - 0.1) * SR)
    t = np.arange(n) / SR
    chord = [110.0, 164.8, 220.0, 277.2, 329.6, 440.0, 554.4]
    pad = sum(np.sin(2 * np.pi * fr * t + r.random() * 6) * (0.9 if fr < 300 else 0.5) for fr in chord)
    pad += sum(0.25 * np.sin(2 * np.pi * fr * 1.003 * t) for fr in chord[3:])
    pad *= np.minimum(t / 0.35, 1) * np.exp(-t / 1.8)
    place(dry, TB + 0.1, pad, gain=0.05)
    place(send, TB + 0.1, pad, gain=0.06)
    n = int(0.6 * SR)
    shing = glide_sine(0.6, 3000, 7000) * env(n, 0.04, 0.15) + sweep_noise(0.6, 6000, 12000, 1.3) * env(n, 0.05, 0.1) * 0.3
    place(dry, 2.95, shing, 0.3, 0.07)
    place(send, 2.95, shing, 0.3, 0.08)
    for _ in range(14):
        n = int(0.25 * SR)
        b = np.sin(2 * np.pi * r.uniform(2500, 6000) * np.arange(n) / SR) * env(n, 0.003, 0.05)
        place(send, TB + 0.2 + r.random() * 1.2, b, r.uniform(-1, 1), 0.04)

    # 殘響
    ir_n = int(2.2 * SR)
    ir_t = np.arange(ir_n) / SR
    irs = [lp(r.standard_normal(ir_n), 6000) * np.exp(-ir_t / 0.55) for _ in range(2)]
    wet = np.stack([signal.fftconvolve(send[:, c], irs[c])[:N] for c in range(2)], 1)
    wet /= np.max(np.abs(wet)) + 1e-9
    mix = dry + wet * np.max(np.abs(dry)) * 0.35
    mix = hp(mix, 22, 2)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix = np.tanh(1.6 * mix) / np.tanh(1.6)
    mix = mix[: int(DUR * SR)]
    fo = int(0.25 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2
    mix *= 0.95 / np.max(np.abs(mix))
    wavfile.write(path, SR, (mix * 32767).astype(np.int16))


def main():
    wav = os.path.splitext(OUT)[0] + "_audio.wav"
    make_audio(wav)
    print("算圖中…", flush=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", "-shortest", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(NF):
        proc.stdin.write(render_frame(f).tobytes())
        if f % 30 == 0:
            print(f"  {f}/{NF}", flush=True)
    proc.stdin.close()
    proc.wait()
    os.remove(wav)
    print("完成：", OUT)


if __name__ == "__main__":
    main()
