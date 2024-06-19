import numpy as np
from connectomics.data.utils.data_io import readvol
from connectomics.utils.process import binary_watershed
from connectomics.utils.evaluate import confusion_matrix, get_binary_jaccard

pred = readvol('outputs/Lucchi_UNet/test/result_txyz.h5')
gt = readvol('datasets/Lucchi/mnt/coxfs01/vcg_connectomics/mitochondria/Lucchi/label/test_label.tif')

pred = binary_watershed(pred)
gt = (gt != 0).astype(np.uint8)
scores = get_binary_jaccard(pred, gt, [0.8])[0]

print("\n=====================")
print(f"Foreground IoU:\t{scores[0]:.3f}\nIoU:\t\t{scores[1]:.3f}\nprecision:\t{scores[2]:.3f}\nrecall:\t\t{scores[3]:.3f}")
print("=====================\n")
