from typing import List, Dict
import json

import matplotlib.pyplot as plt


class BoundingBox:
    def __init__(self, x_center=0, y_center=0, width=0, height=0, iou=0):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.iou = iou


def read_lean_map_of_bboxes(input_json_file_path: str):
    with open(input_json_file_path, 'r') as json_stream:
        raw_object = json.load(json_stream)
    res = {k: [BoundingBox(**item) for item in v]
            for k, v in raw_object.items()}
    return res


def get_minimum_and_maximum_height(boxes, iou):
    passing_boxes_height = []
    for boxes_list in boxes.values():
        height_pass = [box.height for box in boxes_list if box.iou > iou]
        passing_boxes_height.extend(height_pass)
    passing_boxes_min_height = min(passing_boxes_height)
    passing_boxes_max_height = max(passing_boxes_height)
    return passing_boxes_min_height, passing_boxes_max_height


def create_historgram(boxes, iou_threshold):
    title = "iou histogram"
    passing_boxes = get_passing_boxes(boxes, iou_threshold)
    plt.hist(passing_boxes, bins=100)
    plt.xlabel("iou")
    plt.ylabel("#occurrences")
    plt.title(title)
    plt.savefig(title)
    plt.show()


def calculate_average_iou(boxes, iou):
    passing_boxes = get_passing_boxes(boxes, iou)
    return sum(passing_boxes) / len(passing_boxes)


def get_passing_boxes(boxes, iou):
    passing_boxes = []
    for boxes_list in boxes.values():
        iou_pass = [box.iou for box in boxes_list if box.iou > iou]
        passing_boxes.extend(iou_pass)
    return passing_boxes


def get_passing_boxes_height_boundaries(boxes, iou):
    passing_boxes_height = []
    for boxes_list in boxes.values():
        height_pass = [box.height for box in boxes_list if box.iou > iou]
        passing_boxes_height.extend(height_pass)
    passing_boxes_height_min = min(passing_boxes_height)
    passing_boxes_height_max = max(passing_boxes_height)
    return passing_boxes_height_min, passing_boxes_height_max


def calculate_iou_for_2_boxes(box1, box2):
    # assume all numbers are positive
    box1_x_left = box1.x_center - (box1.width / 2)
    box2_x_left = box2.x_center - (box2.width / 2)
    box1_x_right = box1.x_center + (box1.width / 2)
    box2_x_right = box2.x_center + (box2.width / 2)

    box1_y_upper = box1.y_center + (box1.height / 2)
    box2_y_upper = box2.y_center + (box2.height / 2)
    box1_y_lower = box1.y_center - (box1.height / 2)
    box2_y_lower = box2.y_center - (box2.height / 2)

    if box1_x_right < box2_x_left or \
            box2_x_right < box1_x_left or \
            box1_y_lower > box2_y_upper or \
            box2_y_lower > box1_y_upper:
        return 0

    # inner rectangle points
    x_left = max(box1_x_left, box2_x_left)
    x_right = min(box1_x_right, box2_x_right)
    y_upper = min(box1_y_upper, box2_y_upper)
    y_lower = max(box1_y_lower, box2_y_lower)

    intersection = (x_right-x_left) * (y_upper-y_lower)
    union = (box1.width * box1.height) + (box2.width * box2.height) - intersection
    iou = intersection / union
    return iou


if __name__ == '__main__':
    path_detection_boxes_json = "Q1_gt.json"
    path_groundtruth_boxes_json = "Q1_system_output.json"
    iou_threshold = 0.5

    detection_boxes = read_lean_map_of_bboxes(path_detection_boxes_json)
    ground_truth_boxes = read_lean_map_of_bboxes(path_groundtruth_boxes_json)

    for name, detection_bounding_box_list in detection_boxes.items():
        if name not in ground_truth_boxes:
            continue
        ground_truth_bounding_box_list = ground_truth_boxes[name]
        for det_box in detection_bounding_box_list:
            for gt_bbox in ground_truth_bounding_box_list:
                iou = calculate_iou_for_2_boxes(det_box, gt_bbox)

                # determine highest iou for a detection bounding box
                if iou > det_box.iou:
                    det_box.iou = iou

    # calculate average iou for the boxes that pass > iou_threshold
    average_iou = calculate_average_iou(detection_boxes, iou_threshold)
    print("Average iou: {}".format(average_iou))

    # create histogram for the boxes that pass iou_threshold.
    # x_axis: iou, y_axis: occurrences
    create_historgram(detection_boxes, iou_threshold)

    # find the minimum and the maximum height for the boxes that pass > iou_threshold
    min_height, max_height = get_minimum_and_maximum_height(detection_boxes, iou_threshold)
    print("min height: {}, max height: {}".format(min_height, max_height))
