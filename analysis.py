import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def get_img(num):
    img = f'data/disks/IMG_{num}.JPG'
    img = cv2.imread(img)
    return img

def show(img):
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


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