# import necessary libraries
import numpy as np
import SimpleITK as sitk


def sitk_to_numpy(mhd_file, swap=True):
    image = sitk.ReadImage(mhd_file)
    spacing = image.GetSpacing()
    offset = image.GetOrigin()

    image = sitk.GetArrayFromImage(image)
    if swap:
        image = np.swapaxes(image, 0, 2)
    return image, spacing, offset

def numpy_to_sitk(image, spacing, offset, filename, swap=True):
    if swap:
        image = np.swapaxes(image, 0, 2)
    image = sitk.GetImageFromArray(image.astype(np.int16))
    image.SetSpacing(spacing.astype(float))
    image.SetOrigin(offset.astype(float))

    writer = sitk.ImageFileWriter()
    writer.SetUseCompression(True)
    writer.SetFileName(filename)
    writer.Execute(image)
