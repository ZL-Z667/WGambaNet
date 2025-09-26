import os
import numpy as np
import torch
from scipy.ndimage import zoom
import SimpleITK as sitk
from PIL import Image

def calculate_metrics(pred, gt, num_classes):
    metrics = {
        'DSC': 0.0,
        'RV': 0.0,
        'MYO': 0.0,
        'LV': 0.0
    }
    for class_id in range(1, num_classes):
        pred_mask = (pred == class_id)
        gt_mask = (gt == class_id)
        if pred_mask.sum() > 0 or gt_mask.sum() > 0:
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum())
        else:
            dice = 1.0 if (pred_mask.sum() == 0 and gt_mask.sum() == 0) else 0.0
        if class_id == 1:
            metrics['RV'] = dice
        elif class_id == 2:
            metrics['MYO'] = dice
        elif class_id == 3:
            metrics['LV'] = dice
    metrics['DSC'] = (metrics['RV'] + metrics['MYO'] + metrics['LV']) / 3
    return metrics

def test_single_volume(
    image, label, net, classes, patch_size, test_save_path, case, z_spacing, save_slices_dir=None
):
    image = image.cpu().numpy().astype(np.float32) if torch.is_tensor(image) else image.astype(np.float32)
    label = label.cpu().numpy() if torch.is_tensor(label) else label

    if image.ndim == 2:
        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)
    elif image.ndim == 3 and image.shape[0] != 1:
        pass
    else:
        image = np.squeeze(image)
        label = np.squeeze(label)
        if image.ndim == 2:
            image = np.expand_dims(image, 0)
            label = np.expand_dims(label, 0)
    num_slices = image.shape[0]
    prediction = np.zeros_like(label)

    color_map = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    }

    for slice_idx in range(num_slices):
        slice_img = image[slice_idx]
        original_shape = slice_img.shape
        if slice_img.shape != patch_size:
            slice_img = zoom(slice_img,
                             (patch_size[0] / original_shape[0],
                              patch_size[1] / original_shape[1]),
                             order=3)
        input_tensor = torch.from_numpy(slice_img).float().unsqueeze(0).unsqueeze(0).cuda()
        output = net(input_tensor)
        out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
        pred_slice = out.cpu().numpy()
        if pred_slice.shape != original_shape:
            pred_slice = zoom(pred_slice,
                              (original_shape[0] / patch_size[0],
                               original_shape[1] / patch_size[1]),
                              order=0)
        prediction[slice_idx] = pred_slice

        if save_slices_dir is not None:
            os.makedirs(save_slices_dir, exist_ok=True)

            pred_rgb = np.zeros((*pred_slice.shape, 3), dtype=np.uint8)
            for k, color in color_map.items():
                pred_rgb[pred_slice == k] = color
            Image.fromarray(pred_rgb).save(
                os.path.join(save_slices_dir, f"{case}_slice{slice_idx}_pred.png")
            )

            gt_slice = label[slice_idx]
            gt_rgb = np.zeros((*gt_slice.shape, 3), dtype=np.uint8)
            for k, color in color_map.items():
                gt_rgb[gt_slice == k] = color
            Image.fromarray(gt_rgb).save(
                os.path.join(save_slices_dir, f"{case}_slice{slice_idx}_gt.png")
            )

    metrics = calculate_metrics(prediction, label, classes)
    return metrics, prediction