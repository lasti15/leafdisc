import cv2
import numpy as np
#import matplotlib.pyplot as plt
import math
import os
from sklearn.cluster import DBSCAN


def get_img(num):
    img = f'data/disks/IMG_{num}.JPG'
    img = cv2.imread(img)
    return img

# def show(img):
#     plt.imshow(img)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()


def ellipse_eccentricity(ellipse):
    (center_x, center_y), (minor_axis, major_axis), angle = ellipse

    # Ensure minor <= major
    a = max(minor_axis, major_axis)  # major axis
    b = min(minor_axis, major_axis)  # minor axis

    eccentricity = math.sqrt(1 - (b ** 2 / a ** 2))

    return eccentricity

def count_pixels_in_ellipse(ellipse, img_shape, white_mask):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    # Draw filled ellipse on the mask
    center, axes, angle = ellipse
    axes_int = (int(axes[0] / 2), int(axes[1] / 2))  # OpenCV expects semi-axes
    center_int = (int(center[0]), int(center[1]))
    cv2.ellipse(mask, center_int, axes_int, angle, 0, 360, 255, -1)

    total_pixels = cv2.countNonZero(mask)

    white_pixels = cv2.countNonZero(cv2.bitwise_and(mask, white_mask))

    return total_pixels, white_pixels

def count_pixels_in_contour(cnt, img_shape, white_mask):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    # Draw filled contour on the mask
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    white_pixels = cv2.countNonZero(cv2.bitwise_and(mask, white_mask))

    return cv2.contourArea(cnt), white_pixels

def draw_text_in_ellipse(img, ellipse, text, font_scale=0.3, color=(0,0,0), thickness=1):

    center, axes, angle = ellipse
    center_x, center_y = int(center[0]), int(center[1])
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Split text into lines
    lines = text.split('\n')

    # Measure total text height (including spacing)
    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_height = int(max([h for (_, h) in line_sizes])*1.3)
    total_text_height = line_height * len(lines) - 8

    # Starting y so text is vertically centered
    start_y = center_y - total_text_height // 2

    # Draw each line centered horizontally
    for i, (line, (line_width, line_height_val)) in enumerate(zip(lines, line_sizes)):
        x = center_x - line_width // 2
        y = start_y + i * line_height + line_height_val
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def mask_coloured_green(img, hue_min=50, hue_max=170, chroma_min_dark=40, chroma_min_light=80):
    """
    Returns a boolean mask (H, W) that is True where hue is in range and
    chroma exceeds the threshold — stricter for light pixels than dark ones.
    img: RGB uint8 numpy array
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1] - 128
    b = lab[:, :, 2] - 128
    chroma = np.sqrt(a**2 + b**2)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    hue_min_cv = int(hue_min * 179 / 360)
    hue_max_cv = int(hue_max * 179 / 360)

    # L in OpenCV LAB is 0-255 (representing 0-100%), so 50% = 128
    is_light = L > 128
    chroma_threshold = np.where(is_light, chroma_min_light, chroma_min_dark)

    hue_mask = (hue >= hue_min_cv) & (hue <= hue_max_cv)
    chroma_mask = chroma > chroma_threshold

    return hue_mask & chroma_mask


def find_green_hsv_range(image, sample_size=10000,
                         h_lim=(20, 100), s_lim=(25, 255), v_lim=(0, 255),
                         hard_lower=np.array([0, 90, 0]), hard_upper=np.array([80, 255, 200])):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Filter to plausible green region first
    mask = (
            (pixels[:, 0] >= h_lim[0]) & (pixels[:, 0] <= h_lim[1]) &
            (pixels[:, 1] >= s_lim[0]) & (pixels[:, 1] <= s_lim[1]) &
            (pixels[:, 2] >= v_lim[0]) & (pixels[:, 2] <= v_lim[1])
    )
    pixels = pixels[mask]

    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[idx]

    # Normalise before clustering so H/S/V contribute equally
    pixels_norm = pixels.copy()
    pixels_norm[:, 0] /= 179.0
    pixels_norm[:, 1] /= 255.0
    pixels_norm[:, 2] /= 255.0

    db = DBSCAN(eps=0.05, min_samples=50)
    labels = db.fit_predict(pixels_norm)

    # Find most-green cluster (ignoring noise label -1)
    unique_labels = set(labels) - {-1}
    if not unique_labels:
        raise ValueError("DBSCAN found no clusters — try increasing eps or lowering min_samples")

    GREEN_HUE = 60
    best_label = min(unique_labels,
                     key=lambda l: np.abs(pixels[labels == l, 0].mean() - GREEN_HUE))

    green_pixels = pixels[labels == best_label]

    # lower = np.percentile(green_pixels, 1, axis=0).astype(np.uint8)
    # upper = np.percentile(green_pixels, 99, axis=0).astype(np.uint8)

    lower = green_pixels.min(axis=0)
    upper = green_pixels.max(axis=0)

    lower = np.min([lower, hard_lower], axis=0)
    upper = np.max([upper, hard_upper], axis=0)

    return lower, upper, pixels, labels

def align_masks(mask1, mask2):
    # Phase correlation works on float
    m1 = mask1.astype(np.float32)
    m2 = mask2.astype(np.float32)

    shift, response = cv2.phaseCorrelate(m1, m2)
    # shift is (dx, dy) to move mask1 to align with mask2

    dx, dy = shift
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(mask1, M, (mask2.shape[1], mask2.shape[0]))

    return aligned, shift


def crop_around_contour(image, mask, contour, padding=0.20):
    x, y, w, h = cv2.boundingRect(contour)

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    adjusted_contour = contour - np.array([x1, y1])

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], adjusted_contour, np.array([x1, y1])


def alpha_label(n):
    """1-indexed: 1='a', 26='z', 27='aa'"""
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(ord('A') + rem) + result
    return result

def new_analyse_image(img, img_name, radius_default, roundness_limit=0.4, acceptable_radius_variation=0.2, radius_average=[], n_disks=100, chroma_min=15):
    #analyse_image(img, roundness_limit=0.3, acceptable_radius_variation=0.2 , radius_average=[], n_disks=1, radius_default=340, verbose=True)
    img_name = os.path.splitext(img_name)[0]
    # Convert to HSV for easier green thresholding
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    # # Define green range in HSV
    # lower_green, upper_green, _, _ = find_green_hsv_range(hsv, hard_lower=np.array([0, 90, 10]), hard_upper=np.array([90, 255, 255]))
    #
    # mask = cv2.inRange(hsv, lower_green, upper_green)

    mask = (mask_coloured_green(img, chroma_min_dark=5, chroma_min_light=20)*255).astype('uint8')

    if mask.max() == 0:
        return [], img, radius_average

    output_img = img.copy()

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    sorted_indices = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)

    ellipses = []

    n_found = 0

    if len(radius_average) == 0:
        typical_radius = radius_default
    else:
        typical_radius = np.median(radius_average)

    for i in sorted_indices:
        cnt = contours[i]
        if hierarchy[0][i][3] != -1:
            continue

        centre = cnt.mean(axis=0)

        if len(cnt) >= 5 and cv2.contourArea(cnt) > (np.pi * ((typical_radius/20)**2)):  # need at least 5 points to fit ellipse

            n_found += 1
            # if n_found==18:
            #     break

            if n_found >= n_disks:
                break

            filled_cnt = cv2.drawContours(np.zeros_like(mask), [cnt], 0, 255, -1)

            # punch out children
            child_idx = hierarchy[0][i][2]  # first child
            while child_idx != -1:
                child_cnt = contours[child_idx]
                filled_cnt = cv2.drawContours(filled_cnt, [child_cnt], 0, 0, -1)
                child_idx = hierarchy[0][child_idx][0]  # next sibling

            cnt = cv2.convexHull(cnt)
            ellipse = cv2.fitEllipse(cnt)
            roundness = ellipse_eccentricity(ellipse)
            radius = int(np.mean(ellipse[1]))

            if roundness < roundness_limit and radius / typical_radius > (1 - acceptable_radius_variation):
                radius_average.append(radius)
                typical_radius = np.median(radius_average)
                edges_intact = True

            else:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0  # fallback

                center = (cx, cy)
                axes = (typical_radius, typical_radius)
                angle = 0

                ellipse = (center, axes, angle)

                edges_intact = False

            whole_circle = cv2.ellipse(np.zeros_like(mask), ellipse, 255, -1)

            out_whole = cv2.countNonZero(whole_circle)

            erosion_size = 2 * round((typical_radius / 300) / 2) + 1
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))

            dilate_dst = cv2.dilate(filled_cnt, element)
            out_mask = cv2.erode(dilate_dst, element)

            green_sections = cv2.bitwise_and(mask, whole_circle)
            out_mask = cv2.bitwise_or(out_mask, green_sections)

            out_remaining = cv2.countNonZero(out_mask)

            pct = max(round(100 * (1 - (out_remaining / out_whole)), 2), 0)
            label = alpha_label(n_found)
            text =  f"{label}\n\n{pct}%"

            leaf_img = output_img[out_mask>0]
            cv2.ellipse(output_img, ellipse, (254, 0, 0), -1)  # draw yellow ellipse
            output_img[out_mask>0] = leaf_img
            draw_text_in_ellipse(output_img, ellipse, text, font_scale=1, color=(255, 255, 255), thickness=2)

            ellipses.append({'img_name':img_name,
                             'cnt_num':i, 'label':label, 'centre':centre, 'radius':radius, 'edges_intact':edges_intact,
                             'green_size':out_remaining, 'original_size':out_whole, 'pct_missing':pct})


    return ellipses, output_img, radius_average


def analyse_image(img, roundness_limit=0.3, acceptable_radius_variation=0.2 , radius_average=[], n_disks=1, radius_default=340, verbose=True):

    # Convert to HSV for easier green thresholding
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define green range in HSV
    lower_green = np.array([0, 90, 0])  # tweak as needed
    upper_green = np.array([80, 255, 200])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    output_img = img.copy()

    ellipses = []

    n_found = 0

    if len(radius_average) == 0:
        typical_radius = radius_default
    else:
        typical_radius = np.median(radius_average)

    for cnt in contours:
        if len(cnt) >= 5 and cv2.contourArea(cnt)>10000:  # need at least 5 points to fit ellipse

            if n_found>=n_disks:
                break

            cnt = cv2.convexHull(cnt)
            ellipse = cv2.fitEllipse(cnt)
            n_found += 1
            roundness = ellipse_eccentricity(ellipse)
            radius = int(np.mean(ellipse[1]))

            if roundness<roundness_limit and radius/typical_radius > (1-acceptable_radius_variation):
                radius_average.append(radius)
                total, white = count_pixels_in_ellipse(ellipse, output_img.shape, mask)

                pct = f"{round(100*(1-(white/total)),2)}%"

                cv2.ellipse(output_img, ellipse, (254, 209, 0), 3)  # draw green ellipse
                draw_text_in_ellipse(output_img, ellipse, pct, font_scale=2, color=(255, 255, 255), thickness=3)

                if verbose:
                    print(cv2.contourArea(cnt), pct)

                ellipses.append([total, white, None, None])
            else:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0  # fallback

                # Because cv2.ellipse expects (center, axes, angle, startAngle, endAngle, color, thickness)
                # For a circular ellipse, set both axes to (radius, radius)
                center = (cx, cy)
                axes = (typical_radius, typical_radius)
                angle = 0

                ellipse = (center, axes, angle)

                total, white = count_pixels_in_ellipse(ellipse, output_img.shape, mask)
                pct = f"{round(100*(1-(white/total)),1)}%"

                total_cnt, white_cnt = count_pixels_in_contour(cnt, output_img.shape, mask)
                pct_cnt = f"{round(100 * (1 - (white_cnt / total_cnt)), 1)}%"

                label = (f'{pct}\n({pct_cnt})')

                cv2.ellipse(output_img, ellipse, (0, 100, 49), 3)
                cv2.drawContours(output_img, [cnt], 0, (254, 209, 0), 3)
                draw_text_in_ellipse(output_img, ellipse, label, font_scale=2, color=(255, 255, 255), thickness=3)

                if verbose:
                    print(cv2.contourArea(cnt), pct)

                ellipses.append([total, white, total_cnt, white_cnt])


    return ellipses, output_img, radius_average


#analyse_image(img)