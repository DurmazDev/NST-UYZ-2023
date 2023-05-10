import cv2

# This is a sandbox version of KASK.
crop_sizes = [
    [(0, 0), (832, 832)],
    [(544, 0), (1376, 832)],
    [(1088, 0), (1920, 832)],
    [(0, 248), (832, 1080)],
    [(544, 248), (1376, 1080)],
    [(1088, 248), (1920, 1080)]
]

i = '0.jpg'
inner_counter = 0
img = cv2.imread(i)
img_name = i[:-4]
for j in crop_sizes:
    from_x, from_y = j[0]
    to_x, to_y = j[1]
    cropped_image = img[from_y:to_y, from_x:to_x]
    cv2.imwrite(f"{img_name}_{inner_counter}.jpg", cropped_image)
    inner_counter += 1
