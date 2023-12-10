import os
import json

img_folder_path = ["train/", "val/"]
ann_file_path = ["train_labels.json", "val_labels.json"]

for i in range(len(ann_file_path)):
    with open(ann_file_path[i], 'r') as f:
        annotations = json.load(f)
        for image in annotations['images']:
            image_id = image['id']
            image_filename = image['file_name']
            image_width = image['width']
            image_height = image['height']

            # Get the annotations for this image
            image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

            normalized_keypoints = []
            keypoint_values = []
            bbox_values = []
            for annotation in image_annotations:
                keypoints = annotation['keypoints']
                    # normalize x and y coordinates of keypoints

                for i in range(0, len(keypoints), 3):
                    x = keypoints[i]
                    y = keypoints[i+1]
                    visibility = keypoints[i+2]
                    normalized_x = round(x / image_width, 6)  # normalize x coordinate
                    normalized_y = round(y / image_height, 6)  # normalize y coordinate
                    normalized_keypoints.extend([normalized_x, normalized_y, visibility])
                
                for i in range(0, len(keypoints), 3):
                    x = keypoints[i]
                    y = keypoints[i + 1]
                    visibility = keypoints[i + 2]
                    keypoint_values.extend([x, y, visibility])

            # Save the normalized keypoints and bbox to a text file with the same name as the image
            txt_filename = os.path.splitext(image_filename)[0] + '.txt'
            with open(os.path.join("data/labels/" + img_folder_path[i], txt_filename), 'w') as txt_file:
                keypoint_str = ' '.join(map(str, normalized_keypoints))
                txt_file.write(f'0  {keypoint_str}')

print("Processing complete.")