# -*- coding: utf-8 -*-
import numpy as np
from monai.transforms import (
    Orientationd, RandAffined, RandShiftIntensityd, RandGaussianNoised
)

def transform_img_lab(image, label, args):
    data_dicts = {"image": image, "label": label}

    data_dicts = Orientationd(keys=["image", "label"], as_closest_canonical=True)(data_dicts)

    data_dicts = RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=0.3,  # affine 30%
        spatial_size=(args.ROI_shape, args.ROI_shape),
        translate_range=(15, 15),
        rotate_range=np.pi / 36,
        scale_range=(0.05, 0.05),
        padding_mode="zeros"
    )(data_dicts)

    data_dicts = RandShiftIntensityd(
        keys=["image"], offsets=0.1, prob=0.3)(data_dicts)

    data_dicts = RandGaussianNoised(
        keys=["image"], prob=0.3, mean=0.0, std=0.05)(data_dicts)

    return data_dicts



# # -*- coding: utf-8 -*-
# import numpy as np
# import random
# from monai.transforms import (
#     Orientationd, RandAffined, ScaleIntensityRanged, RandGaussianNoised
# )

# def transform_img_lab(image, label, args):
#     data_dicts = {"image": image, "label": label}

#     # 1) 方向标准化
#     orientation = Orientationd(keys=["image", "label"], as_closest_canonical=True)

#     # 2) 轻度仿射（平移+旋转+缩放）
#     rand_affine = RandAffined(
#         keys=["image", "label"],
#         mode=("bilinear", "nearest"),
#         prob=1.0,
#         spatial_size=(args.ROI_shape, args.ROI_shape),
#         translate_range=(15, 15),
#         rotate_range=(np.pi / 36, np.pi / 36),
#         scale_range=(0.05, 0.05),
#         padding_mode="zeros"
#     )

#     # 3) 灰度缩放（亮度变化）
#     scale_shift = ScaleIntensityRanged(
#         keys=["image"], a_min=-10, a_max=10, b_min=-10, b_max=10, clip=True
#     )

#     # 4) 轻噪声
#     gauss_noise = RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=1)

#     # --- 简化触发逻辑 ---
#     if random.random() < 0.3:  # 50% 进行方向标准化
#         data_dicts = orientation(data_dicts)

#     if random.random() < 0.2:  # 70% 进行仿射
#         data_dicts = rand_affine(data_dicts)

#     if random.random() < 0.3:  # 50% 灰度范围缩放
#         data_dicts = scale_shift(data_dicts)

#     if random.random() < 0.2:  # 30% 轻噪声
#         data_dicts = gauss_noise(data_dicts)

#     return data_dicts
