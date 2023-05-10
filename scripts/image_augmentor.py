import os
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import imgaug as ia
import multiprocessing
from tqdm import tqdm


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (float(box[0]) + float(box[1]))/2.0
    y = (float(box[2]) + float(box[3]))/2.0
    w = float(box[1]) - float(box[0])
    h = float(box[3]) - float(box[2])
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def unconvert(width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    return (xmin, ymin, xmax, ymax)


# Veri dizinindeki tüm resim ve annotation dosyalarını yükleme
def multip(image_file):
    # Resmi yükleme
    data_dir = "images"
    output_dir = "augmented"
    new_size = (384, 384)

    def sometimes(aug): return iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate by -20 to +20 percent (per axis)
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                # use nearest neighbour or bilinear interpolation (fast)
                order=[0, 1],
                # if mode is constant, use a cval between 0 and 255
                cval=(0, 255),
                # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                mode=ia.ALL
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                # convert images into their superpixel representation
                iaa.OneOf([
                       # blur images with a sigma between 0 and 3.0
                       iaa.GaussianBlur((0, 3.0)),
                       # blur image using local means with kernel sizes between 2 and 7
                       iaa.AverageBlur(k=(2, 7)),
                       # blur image using local medians with kernel sizes between 2 and 7
                       iaa.MedianBlur(k=(3, 11)),
                       ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(
                    0.75, 1.5)),  # sharpen images
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.05*255), per_channel=0.5),
                iaa.OneOf([
                    # randomly remove up to 10% of the pixels
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(
                        0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True),  # invert color channels
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                # improve or worsen the contrast
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                # move pixels locally around (with random strengths)
                sometimes(iaa.ElasticTransformation(
                    alpha=(0.5, 3.5), sigma=0.25)),
                # sometimes move pa rts of the image around
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
                random_order=True
            )
        ],
        random_order=True
    )
    image_path = os.path.join(data_dir, image_file)
    image = imageio.imread(image_path)

    # Annotation dosyasını yükleme
    annotation_path = os.path.join(
        data_dir, os.path.splitext(image_file)[0] + ".txt")
    with open(annotation_path, "r") as f:
        annotation_lines = f.readlines()

    # Bounding box'ları ayrıştırma
    bboxes = []
    for line in annotation_lines:
        class_id, x, y, w, h = map(float, line.strip().split())
        xmin, ymin, xmax, ymax = unconvert(
            new_size[0], new_size[1], x, y, w, h)
        bbox = BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=class_id)
        bboxes.append(bbox)

    # Resim ve bounding box'ları artırma
    bbs = BoundingBoxesOnImage(bboxes, shape=image.shape)
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # Artırılmış resim ve bounding box'ları kaydetme
    image_aug_path = os.path.join(output_dir, "augmented_" + image_file)
    imageio.imwrite(image_aug_path, image_aug)

    bbs_aug_path = os.path.join(
        output_dir, "augmented_" + os.path.splitext(image_file)[0] + ".txt")
    with open(bbs_aug_path, "w") as f:
        for bbox in bbs_aug:
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            x, y, w, h = convert(new_size, (x1, x2, y1, y2))
            class_id = bbox.label
            f.write(f"{int(class_id)} {x} {y} {w} {h}\n")


if __name__ == '__main__':
    # Veri dizini yolu

    data_dir = "images"
    image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
    # imgaug augmentasyon işlemleri
    pool = multiprocessing.Pool()
    num_files = len(image_files)
    with tqdm(total=num_files) as pbar:
        for _ in pool.imap_unordered(multip, image_files):
            pbar.update()

    pool.close()
    pool.join()
    print("Finished...")
