# Group: Emilio Brambilla, Lasse Haffke, Moritz Lahann

import scipy.io as scio
import numpy as np


def undersegmentation(ground_truth, superpixels, label):
    # Get the labels in the SLIC image that overlap with the current ground truth label
    # print("打印1：",[ground_truth == label])
    # print(superpixels)
    overlapping_superpixels = superpixels[ground_truth == label]
    # print("打印2：",overlapping_superpixels)
    # print("overlapping_superpixels:",overlapping_superpixels)
    superpixel_labels = np.unique(overlapping_superpixels)

    # Count the number of pixels in the SLIC image that are labeled with an overlapping label from above
    superpixel_area = len(superpixels[np.isin(superpixels, superpixel_labels)])

    # Count the number of pixels in the ground truth with the current label
    ground_truth_area = len(ground_truth[ground_truth == label])

    # Calculate undersegmentation error
    return (superpixel_area - ground_truth_area) / ground_truth_area,superpixel_area,ground_truth_area


if __name__ == "__main__":

    # input_image = scio.loadmat("评价指标/before1_850_55.mat")['labels']
    input_image = scio.loadmat("评价指标/before1_1500_55.mat")['labels']
    ground_truth = scio.loadmat("评价指标/before1_GT.mat")['I1']

    # Run SLIC with skimage default parameters (n=100, c=10.0)
    errors = []
    superpixel_counts = []
    ground_truth_counts = []

    # 检索GT中的每个标签的序号，进行遍历，与原图片比较
    for label in np.unique(ground_truth):
        if label > 0:
            # Calculate undersegmentation error of label
            error,superpixel_count, ground_truth_count = undersegmentation(ground_truth, input_image, label)
            errors.append(error)
            superpixel_counts.append(superpixel_count)
            ground_truth_counts.append(ground_truth_count)

    # Calculate average error over all labels in ground truth
    # #  {np.mean(errors)}求的不知道是什么
    # print(f"The average undersegmentation error is {np.mean(errors)}.")
    fin_count = (np.sum(superpixel_counts) - np.sum(ground_truth_counts)) / np.sum(ground_truth_counts)
    print("Undersegmentation Error :",fin_count*100,"%")
