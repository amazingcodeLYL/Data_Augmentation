import albumentations as A
from albumentations.pytorch.transforms import ToTensor


def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),


            A.Rotate(limit=30, p=0.5),
            A.CenterCrop(512, 512, p=0.5),
            A.PadIfNeeded(800, 800, p=0.5),
            # A.RandomContrast(limit=0.2, p=0.5),
            A.MotionBlur(p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.CLAHE(p=0.5),
            # A.JpegCompression(p=0.5),
            A.ImageCompression(p=0.5),
            A.RandomRain(p=0.5),
            A.RandomFog(p=0.5),
            A.RandomShadow(p=0.5),


            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )