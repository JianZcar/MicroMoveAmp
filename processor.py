from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np


def load_frames(file_storages):
    # Check type of first element
    if len(file_storages) > 0 and isinstance(file_storages[0], np.ndarray):
        # Already numpy arrays, just return them
        return file_storages

    with ThreadPoolExecutor() as executor:
        for f in file_storages:
            f.seek(0)
        return list(executor.map(read_file_storage, file_storages))
        

def compute_optical_flow(frames, amplify_factor=2.5):
    h, w = frames[0].shape
    motion_magnitude = np.zeros((h, w), dtype=np.float32)
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    for i in range(1, len(frames)):
        flow = optical_flow.calc(frames[i - 1], frames[i], None)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitude += mag

    motion_magnitude = np.power(motion_magnitude, amplify_factor)
    motion_magnitude = cv2.normalize(motion_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return motion_magnitude.astype(np.uint8)


def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


@njit(parallel=True, fastmath=True)
def _boost_blocks(activity_map, grid_x, grid_y, grid_size, threshold, boost):
    h, w = activity_map.shape
    boosted = activity_map.astype(np.float32)

    min_x = grid_x.min()
    max_x = grid_x.max()
    min_y = grid_y.min()
    max_y = grid_y.max()

    for gx in range(min_x, max_x + 1, grid_size):
        for gy in range(min_y, max_y + 1, grid_size):
            sum_val = 0.0
            count = 0
            for i in prange(h):
                for j in range(w):
                    if grid_x[i, j] == gx and grid_y[i, j] == gy:
                        sum_val += activity_map[i, j]
                        count += 1
            if count > 0:
                mean_val = sum_val / count
                if mean_val > threshold:
                    for i in prange(h):
                        for j in range(w):
                            if grid_x[i, j] == gx and grid_y[i, j] == gy:
                                boosted[i, j] *= boost

    return np.clip(boosted, 0, 255).astype(np.uint8)


def grid_micro_motion_boost_rotated(activity_map, grid_size=16, boost=1.6, angle_deg=30):
    h, w = activity_map.shape
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    Y, X = np.indices((h, w))
    cx, cy = w // 2, h // 2
    Xc, Yc = X - cx, Y - cy

    Xr = (Xc * cos_a + Yc * sin_a).astype(np.int32)
    Yr = (-Xc * sin_a + Yc * cos_a).astype(np.int32)

    grid_x = (Xr // grid_size) * grid_size
    grid_y = (Yr // grid_size) * grid_size

    return _boost_blocks(activity_map, grid_x, grid_y, grid_size, threshold=10, boost=boost)


def detect_pixelation_motion_multistage(frames, downscale_factors=[4, 8, 12]):
    h, w = frames[0].shape
    accum = np.zeros((h, w), dtype=np.float32)

    for factor in downscale_factors:
        for i in range(1, len(frames)):
            small_prev = cv2.resize(frames[i - 1], (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)
            small_curr = cv2.resize(frames[i], (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)
            diff = cv2.absdiff(small_prev, small_curr)
            diff = cv2.resize(diff, (w, h), interpolation=cv2.INTER_LINEAR)
            accum += diff

    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)
    accum = cv2.GaussianBlur(accum, (7, 7), 0)
    return accum.astype(np.uint8)


def sharpen_image(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


def amplify_and_show_big_changes(activity_map, base_img, macro_thresh=50):
    micro_heatmap = cv2.applyColorMap(activity_map, cv2.COLORMAP_JET)
    _, macro_mask = cv2.threshold(activity_map, macro_thresh, 255, cv2.THRESH_BINARY)
    macro_heatmap = np.zeros_like(micro_heatmap)
    macro_heatmap[:, :, 0] = macro_mask
    base_bgr = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    blend_micro = cv2.addWeighted(micro_heatmap, 0.7, base_bgr, 0.3, 0)
    final = cv2.addWeighted(blend_micro, 1.0, macro_heatmap, 0.5, 0)
    return final


def detect_edges(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 1)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def process_frames(paths, out_vis_path=None):
    frames = load_frames(paths)
    motion_map = compute_optical_flow(frames, amplify_factor=2.5)
    pixelated_map = detect_pixelation_motion_multistage(frames)

    activity = cv2.addWeighted(motion_map, 0.6, pixelated_map, 0.4, 0)
    activity = cv2.GaussianBlur(activity, (3, 3), 0)
    _, activity = cv2.threshold(activity, 10, 255, cv2.THRESH_TOZERO)

    activity = grid_micro_motion_boost_rotated(activity, grid_size=12, boost=1.7, angle_deg=30)
    activity = cv2.normalize(activity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mid_idx = len(frames) // 2
    base = enhance_contrast(frames[mid_idx])

    final_vis = amplify_and_show_big_changes(activity, base)
    final_vis = sharpen_image(final_vis)

    edges = detect_edges(base)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[edges > 0] = [0, 0, 180]

    final_vis = cv2.addWeighted(final_vis, 1.0, edges_colored, 0.5, 0)

    return final_vis
