import cv2

def calculate_pct_missing(row):
    full_mask = cv2.imread(row['leaf'], 0)
    dmg_mask = cv2.imread(row['damage'], 0)

    full_leaf = full_mask.sum().item()
    missing = dmg_mask.sum().item()

    if full_leaf == 0:
        return full_leaf, missing, 0

    return missing/full_leaf