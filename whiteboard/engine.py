import os

import cv2
import numpy as np
import math

def euc_dist(arr1, point):
    """
    Calculates the Euclidean distance between a set of points and a reference point.
    """
    square_sub = (arr1 - point) ** 2
    return np.sqrt(np.sum(square_sub, axis=1))

def preprocess_image(img_path, resize_wd, resize_ht):
    """
    Reads an image, resizes it, and generates a binary adaptive threshold mask.
    The threshold mask is used to identify "ink" pixels for the animation.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None, None
    img_ht, img_wd = img.shape[0], img.shape[1]
    img = cv2.resize(img, (resize_wd, resize_ht))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian adaptive thresholding: identifies edges and details
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    return img, img_thresh, img_ht, img_wd, img_gray

def preprocess_hand_image(hand_path, hand_mask_path):
    """
    Prepares the 'drawing hand' asset by cropping it to its visible area 
    and preparing foreground/background masks for blending.
    """
    hand = cv2.imread(hand_path)
    hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
    if hand is None or hand_mask is None:
        return None, None, None, None, None

    # Get extreme coordinates to crop hand to its mask size (removes unused empty space)
    indices = np.where(hand_mask == 255)
    x, y = indices[1], indices[0]
    top_left = (np.min(x), np.min(y))
    bottom_right = (np.max(x), np.max(y))
    
    hand = hand[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    hand_mask = hand_mask[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    hand_mask_inv = 255.0 - hand_mask

    # Normalize masks to [0.0, 1.0] for multiplicative blending
    hand_mask = hand_mask / 255.0
    hand_mask_inv = hand_mask_inv / 255.0

    # Ensure hand background is blacked out
    hand_bg_ind = np.where(hand_mask == 0)
    hand[hand_bg_ind] = [0, 0, 0]

    hand_ht, hand_wd = hand.shape[0], hand.shape[1]
    return hand, hand_mask, hand_mask_inv, hand_ht, hand_wd

def draw_hand_on_img(drawing, hand, drawing_coord_x, drawing_coord_y, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd):
    """
    Blends the hand image onto the current frame at the specified coordinates.
    Handles boundaries to prevent cropping/overflow errors.
    """
    remaining_ht = img_ht - drawing_coord_y
    remaining_wd = img_wd - drawing_coord_x
    crop_hand_ht = min(hand_ht, remaining_ht)
    crop_hand_wd = min(hand_wd, remaining_wd)

    if crop_hand_ht <= 0 or crop_hand_wd <= 0:
        return drawing

    # Crop hand if it goes off the bottom/right edge
    hand_cropped = hand[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]

    # Perform blending: black out the background area then add the hand foreground
    for c in range(3):
        drawing[drawing_coord_y : drawing_coord_y + crop_hand_ht, drawing_coord_x : drawing_coord_x + crop_hand_wd, c] = (
            drawing[drawing_coord_y : drawing_coord_y + crop_hand_ht, drawing_coord_x : drawing_coord_x + crop_hand_wd, c] * hand_mask_inv_cropped
        )
    
    drawing[drawing_coord_y : drawing_coord_y + crop_hand_ht, drawing_coord_x : drawing_coord_x + crop_hand_wd] += hand_cropped
    return drawing

def _is_letter_component(stat, img_ht):
    """A lone glyph, such as the label x, rather than a sketch fragment."""
    _x, _y, width, height, area = [int(v) for v in stat]
    if area < 80 or height < 0.02 * img_ht or height > 0.12 * img_ht:
        return False
    shorter = max(1, min(width, height))
    if max(width, height) / shorter > 2.4:
        return False
    return area / float(width * height) >= 0.28

def _is_text_line(component_ids, stats):
    """Several glyphs sharing a baseline, such as a word or a formula."""
    if len(component_ids) < 3:
        return False
    heights = np.array([int(stats[i][3]) for i in component_ids], dtype=float)
    bottoms = np.array([int(stats[i][1]) + int(stats[i][3]) for i in component_ids], dtype=float)
    median_height = float(np.median(heights))
    body = heights >= median_height * 0.75
    if int(body.sum()) < 3:
        return False
    baseline = float(np.median(bottoms[body]))
    aligned = np.abs(bottoms[body] - baseline) <= max(8.0, 0.45 * median_height)
    return int(aligned.sum()) >= 3

def _find_text_lines(ink_mask, img_ht):
    """
    Group letter-like ink into reading-order lines.
    Returns (lines, text_ids, labels). Each line is a list of component ids
    sorted left to right. Stroke-like ink is left out of text_ids.
    """
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    max_side = max(24, int(0.12 * img_ht))
    candidates = []
    for index in range(1, count):
        _x, _y, width, height, area = [int(v) for v in stats[index]]
        if area < 25 or max(width, height) > max_side or min(width, height) < 2:
            continue
        if area / float(width * height) < 0.18:
            continue
        candidates.append(index)
    if not candidates:
        return [], [], labels

    def center_y(index):
        top, height = int(stats[index][1]), int(stats[index][3])
        return top + height / 2.0

    median_height = float(np.median([int(stats[i][3]) for i in candidates]))
    # Anchor each band to its first glyph so a chain of marks cannot walk
    # down the page and swallow the next line. The gap is wide enough to
    # keep superscripts with the formula they belong to.
    band_gap = max(1.4 * median_height, 0.07 * img_ht)
    # Handwritten lines slope, and superscripts sit above the baseline.
    # Keep a glyph in the band when it is still close to that band's first
    # glyph. The anchor does not move, so lines cannot chain down the page.
    span_limit = max(band_gap, 0.08 * img_ht)
    word_gap = 4.5 * median_height
    ordered = sorted(candidates, key=center_y)
    bands = [[ordered[0]]]
    for index in ordered[1:]:
        anchor = center_y(bands[-1][0])
        if center_y(index) - anchor > span_limit:
            bands.append([index])
        else:
            bands[-1].append(index)

    groups = []
    for band in bands:
        band.sort(key=lambda index: int(stats[index][0]))
        current = [band[0]]
        for index in band[1:]:
            previous = current[-1]
            prev_right = int(stats[previous][0]) + int(stats[previous][2])
            gap = int(stats[index][0]) - prev_right
            if gap > word_gap:
                groups.append(current)
                current = [index]
            else:
                current.append(index)
        groups.append(current)

    lines = []
    singles = []
    for component_ids in groups:
        if _is_text_line(component_ids, stats):
            lines.append(component_ids)
        elif len(component_ids) == 1 and _is_letter_component(stats[component_ids[0]], img_ht):
            singles.append(component_ids[0])

    # Superscripts sit just above a formula line. Keep them with that line
    # so they are written after the letter they belong to, not as a separate row.
    unattached = []
    for single in singles:
        sx, sy, sw, sh, _area = [int(v) for v in stats[single]]
        single_bottom = sy + sh
        attached_to = None
        for line in lines:
            line_top = min(int(stats[i][1]) for i in line)
            # Only a superscript sitting just above this line. A label farther
            # up the page must stay on its own line.
            gap_y = line_top - single_bottom
            if not (-0.25 * sh <= gap_y <= max(12, 0.9 * sh)):
                continue
            for member in line:
                mx, _my, mw, _mh, _area = [int(v) for v in stats[member]]
                gap = max(0, max(sx, mx) - min(sx + sw, mx + mw))
                if gap <= sw:
                    attached_to = line
                    break
            if attached_to is not None:
                break
        if attached_to is not None:
            attached_to.append(single)
        else:
            unattached.append([single])
    lines.extend(unattached)

    for line in lines:
        line.sort(key=lambda index: (int(stats[index][0]), int(stats[index][1])))
    lines.sort(key=lambda line: (
        min(int(stats[index][1]) for index in line),
        min(int(stats[index][0]) for index in line),
    ))
    text_ids = [index for line in lines for index in line]
    return lines, text_ids, labels

def _glyph_mask(labels, component_id, gray):
    """Letter pixels plus a little antialiasing, without grabbing a neighbor."""
    component = labels == component_id
    rows, cols = np.where(component)
    if len(rows) == 0:
        return component
    y0 = max(0, int(rows.min()) - 2)
    y1 = min(gray.shape[0], int(rows.max()) + 3)
    x0 = max(0, int(cols.min()) - 2)
    x1 = min(gray.shape[1], int(cols.max()) + 3)
    local = component[y0:y1, x0:x1].astype(np.uint8)
    local = cv2.dilate(local, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    local &= gray[y0:y1, x0:x1] < 250
    painted = np.zeros(component.shape, dtype=bool)
    painted[y0:y1, x0:x1] = local
    return painted

def _write_glyph(drawn_frame, img_orig, glyph, video_object, hand, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd):
    """Reveal one glyph left to right. Two frames for a letter, one for a dot."""
    rows, cols = np.where(glyph)
    if len(cols) == 0:
        return
    x0, x1 = int(cols.min()), int(cols.max())
    if x1 - x0 < 10:
        cuts = [x1]
    else:
        cuts = [x0 + (x1 - x0) // 2, x1]
    for cut in cuts:
        reveal = glyph.copy()
        reveal[:, cut + 1 :] = False
        drawn_frame[reveal] = img_orig[reveal]
        column_rows = np.where(glyph[:, cut])[0]
        tip_y = int(np.median(column_rows)) if len(column_rows) else int(np.median(rows))
        frame = draw_hand_on_img(
            drawn_frame.copy(), hand, cut, tip_y, hand_mask_inv,
            hand_ht, hand_wd, img_ht, img_wd
        )
        video_object.write(frame)

def _write_text_lines(
    lines, labels, gray, drawn_frame, img_orig, video_object,
    hand, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd
):
    for line in lines:
        for component_id in line:
            glyph = _glyph_mask(labels, component_id, gray)
            _write_glyph(
                drawn_frame, img_orig, glyph, video_object, hand, hand_mask_inv,
                hand_ht, hand_wd, img_ht, img_wd
            )

def draw_masked_object(
    drawn_frame, img_thresh, img_orig, video_object, hand, hand_mask_inv, hand_ht, hand_wd,
    resize_ht, resize_wd, split_len, object_mask=None, skip_rate=5, black_pixel_threshold=10
):
    """
    Core animation logic: simulates drawing a specific part of the image (or the whole image).
    Strokes are visited in nearest-neighbor grid order. Text is detected separately and
    written left to right, one glyph at a time, in the original letter colors.
    """
    img_thresh_copy = img_thresh.copy()
    object_ind = None
    if object_mask is not None:
        # If a mask is provided, ignore pixels outside the mask
        object_mask_black_ind = np.where(object_mask == 0)
        object_ind = np.where(object_mask == 255)
        img_thresh_copy[object_mask_black_ind] = 255

    ink_mask = ((img_thresh_copy < black_pixel_threshold).astype(np.uint8)) * 255
    text_lines, text_ids, labels = _find_text_lines(ink_mask, resize_ht)
    if text_ids:
        img_thresh_copy[np.isin(labels, text_ids)] = 255
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    hand_args = (hand, hand_mask_inv, hand_ht, hand_wd, resize_ht, resize_wd)

    stroke_rows = np.where(img_thresh_copy < black_pixel_threshold)[0]
    stroke_top = int(stroke_rows.min()) if len(stroke_rows) else resize_ht
    text_before = []
    text_after = []
    for line in text_lines:
        line_bottom = max(int(np.where(labels == index)[0].max()) for index in line)
        if line_bottom < stroke_top - 4:
            text_before.append(line)
        else:
            text_after.append(line)

    _write_text_lines(
        text_before, labels, gray, drawn_frame, img_orig, video_object, *hand_args
    )

    n_cuts_vertical = int(math.ceil(resize_ht / split_len))
    n_cuts_horizontal = int(math.ceil(resize_wd / split_len))

    # Padding to ensure the image fits the grid exactly
    pad_h = n_cuts_vertical * split_len - resize_ht
    pad_w = n_cuts_horizontal * split_len - resize_wd
    if pad_h > 0 or pad_w > 0:
        img_thresh_copy = cv2.copyMakeBorder(img_thresh_copy, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)

    # Split image into a grid of small blocks
    grid_of_cuts = np.array(np.split(img_thresh_copy, n_cuts_horizontal, axis=-1))
    grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))

    # Find which grid blocks contain 'ink' pixels
    cut_having_black = (grid_of_cuts < black_pixel_threshold) * 1
    cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
    cut_black_indices = np.array(np.where(cut_having_black > 0)).T

    selected_ind = 0
    counter = 0
    while len(cut_black_indices) > 0:
        # Get coordinates of the current grid block
        selected_ind_val = cut_black_indices[selected_ind].copy()
        range_v_start = selected_ind_val[0] * split_len
        range_v_end = range_v_start + split_len
        range_h_start = selected_ind_val[1] * split_len
        range_h_end = range_h_start + split_len

        # Map back to original image size (clipping padding)
        v_s, v_e = range_v_start, min(range_v_end, resize_ht)
        h_s, h_e = range_h_start, min(range_h_end, resize_wd)
        
        if v_s < resize_ht and h_s < resize_wd:
            # Transfer the ink to the 'drawn_frame'
            block = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]][:v_e-v_s, :h_e-h_s]
            if object_mask is not None:
                # Only update pixels that belong to this object's mask (255)
                # This prevents the background pass from overwriting already colored objects
                m_slice = object_mask[v_s:v_e, h_s:h_e] == 255
                for c in range(3):
                   drawn_frame[v_s:v_e, h_s:h_e, c][m_slice] = block[m_slice]
            else:
                for c in range(3):
                    drawn_frame[v_s:v_e, h_s:h_e, c] = block

        # Position for the drawing hand
        hand_coord_x = h_s + int((h_e - h_s) / 2)
        hand_coord_y = v_s + int((v_e - v_s) / 2)
        
        # Remove the block we just finished drawing
        cut_black_indices = np.delete(cut_black_indices, selected_ind, axis=0)
        
        # Select the NEXT block to draw based on proximity (nearest neighbor)
        if len(cut_black_indices) > 0:
            euc_arr = euc_dist(cut_black_indices, selected_ind_val)
            selected_ind = np.argmin(euc_arr)

        counter += 1
        # Periodically write a frame to the video to create the animation effect
        if counter % skip_rate == 0:
            frame_to_write = draw_hand_on_img(
                drawn_frame.copy(), hand, hand_coord_x, hand_coord_y, hand_mask_inv,
                hand_ht, hand_wd, resize_ht, resize_wd
            )
            video_object.write(frame_to_write)

    _write_text_lines(
        text_after, labels, gray, drawn_frame, img_orig, video_object, *hand_args
    )

    # After the animation pass, fill the area with the full-color original pixels
    if object_mask is not None:
        drawn_frame[object_ind] = img_orig[object_ind]
    else:
        drawn_frame[:] = img_orig[:]


def _load_bgr(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _fit_size(image, width, height):
    source_h, source_w = image.shape[:2]
    if width is None and height is None:
        return source_w, source_h
    if width is None:
        width = max(1, int(round(source_w * (height / source_h))))
    elif height is None:
        height = max(1, int(round(source_h * (width / source_w))))
    return width, height


def _threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )


def _new_ink_mask(previous, current):
    """Pixels this step adds. Steps are cumulative pictures of the drawing so far."""
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    if previous is None:
        changed = gray < 250
    else:
        delta = np.abs(current.astype(np.int16) - previous.astype(np.int16)).max(axis=2)
        changed = (delta > 18) & (gray < 250)
    mask = changed.astype(np.uint8) * 255
    if mask.any():
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def animate(
    image_paths,
    output_path,
    frame_rate=25,
    width=None,
    height=None,
    split_len=10,
    skip_rate=7,
    end_seconds=1,
):
    """
    Render a whiteboard video.

    One image is drawn in a single pass. Several images are treated as
    cumulative steps: each frame already contains the previous drawing, and
    only the new ink is animated.
    """
    if not image_paths:
        raise ValueError("At least one image is required.")

    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    hand, _hand_mask, hand_mask_inv, hand_ht, hand_wd = preprocess_hand_image(
        os.path.join(assets_dir, "drawing-hand.png"),
        os.path.join(assets_dir, "hand-mask.png"),
    )
    if hand is None:
        raise FileNotFoundError(f"Hand assets not found in {assets_dir}")

    frames = [_load_bgr(path) for path in image_paths]
    width, height = _fit_size(frames[0], width, height)
    frames = [
        frame if frame.shape[1] == width and frame.shape[0] == height else cv2.resize(frame, (width, height))
        for frame in frames
    ]

    output_path = os.path.abspath(output_path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {output_path}")

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    previous = None
    multiple = len(frames) > 1
    for frame in frames:
        mask = _new_ink_mask(previous, frame) if multiple else None
        if mask is not None and not np.any(mask):
            previous = frame
            continue
        draw_masked_object(
            canvas, _threshold(frame), frame, writer,
            hand, hand_mask_inv, hand_ht, hand_wd,
            height, width, split_len,
            object_mask=mask, skip_rate=skip_rate,
        )
        previous = frame

    for _ in range(int(frame_rate * end_seconds)):
        writer.write(frames[-1])
    writer.release()
    return output_path

