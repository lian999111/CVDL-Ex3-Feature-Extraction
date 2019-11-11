import cv2
import numpy as np

def compute_hog(cell_size, block_size, nbins, imgs_gray):
    """
    Wrapper for the OpenCV interface for HOG features.

    :param cell_size  number of pixels in a square cell in x and y direction (e.g. (4,4), (8,8))
    :param block_size number of cells in a block in x and y direction (e.g., (1,1), (1,2))
    :param nbins      number of bins in a orientation histogram in x and y direction (e.g. 6, 9, 12)
    :param imgs_gray  images with which to perform HOG feature extraction (dimensions (nr, width, height))
    :return array of shape H x imgs_gray.shape[0] where H is the size of the resulting HOG feature vector
            (depends on parameters)
    """
    hog = cv2.HOGDescriptor(_winSize=(imgs_gray.shape[2] // cell_size[1] * cell_size[1],
                                      imgs_gray.shape[1] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    # winSize is the size of the image cropped to a multiple of the cell size

    hog_example = hog.compute(np.squeeze(imgs_gray[0, :, :]).astype(np.uint8)).flatten().astype(np.float32)

    hog_feats = np.zeros([imgs_gray.shape[0], hog_example.shape[0]])

    for img_idx in range(imgs_gray.shape[0]):
        hog_image = hog.compute(np.squeeze(imgs_gray[img_idx, :, :]).astype(np.uint8)).flatten().astype(np.float32)
        hog_feats[img_idx, :] = hog_image

    return hog_feats