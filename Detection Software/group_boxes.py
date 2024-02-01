# This function is responsible to group overlaped boxes to simplify the visualization
def group_boxes(boxes):
    boxes_agrup = []  # boxes agrupation list
    if len(boxes) > 0:  # If has detections, start the group analysis process

        # Data dictionary
        data = {
            'x_init': [],
            'y_init': [],
            'x_end': [],
            'y_end': [],
            'box_id': [],
            'grouped': 0,
            'confidence': []
        }
        # Converting the dictionary into a PD Dataframe
        boxes_agrup = pd.DataFrame(data)

        # Getting a list os unique labels (classes) present in "boxes"
        labels = boxes['box_id'].unique().tolist()

        # Iterating on labels
        for label in labels:
            # Extrating all "equal label" in each "labels" iterations
            # If we have a labels list with "orange, grape, orange",
            # as a result we will have a list with "orange, orange"
            # in the first iteration and "grape" in the second iteration
            filtered = boxes[boxes['box_id'] == label]  # Filtering "label" on "boxes"
            overlap_boxes = []  # overlap list
            confidence_values = []  # Confidence of each detection list

            # Iterating on the filtered labels
            for i in range(1,
                           len(filtered)):  # Geting the length to iterate on each "filtered" element, sraering in 1
                for j in range(i):  # Iterating in range of i, starting in 0 to realize a combinational analysis
                    box_i = filtered.iloc[i]  # Getting the "i", that is the START position of the ROI
                    box_j = filtered.iloc[j]  # gerring the "j", that is the END position of the ROI

                    # Overlap analysis - Verifying if has any overlap between the start/end of each "filtered" box
                    overlap = (
                            (box_i['x_end'] > box_j['x_init']) and (box_i['x_init'] < box_j['x_end']) and
                            (box_i['y_end'] > box_j['y_init']) and (box_i['y_init'] < box_j['y_end'])
                    )

                    # If any overlap is detected
                    if overlap:
                        # Extending "overlap_boxes" list with the position of the overlaped ROI
                        overlap_boxes.extend([box_i, box_j])
                        # Storing the confidence for future calculation
                        confidence_values.extend([box_i['confidence'], box_j['confidence']])

            # If any overlaped boxes is detected
            if overlap_boxes:
                # Find the minimum and maximum coordinates to span all overlapping boxes
                # Here, we are using min/max to get the min/max values,
                # returning the x_init/y_init through a lambda function called "x"
                # As a result, min/max will find dictnary with the min/max, for specified key (x_init, y_init etc)
                # and we need to get the specific value
                x_init_min = min(overlap_boxes, key=lambda x: x['x_init'])['x_init']  # Getting the x_init min
                y_init_min = min(overlap_boxes, key=lambda x: x['y_init'])['y_init']  # Getting the y_init min
                x_end_max = max(overlap_boxes, key=lambda x: x['x_end'])['x_end']  # Getting the x_init max
                y_end_max = max(overlap_boxes, key=lambda x: x['y_end'])['y_end']  # Getting the y_init max

                # Calculate the average of confidences of the overlapped boxes
                mean_confidence = sum(confidence_values) / len(confidence_values)

                # Concatenating  the results into a PD Dataframe to have a list of grouped boxes
                boxes_agrup = pd.concat([boxes_agrup,
                                         pd.DataFrame(
                                             [{'x_init': x_init_min,
                                               'y_init': y_init_min,
                                               'x_end': x_end_max,
                                               'y_end': y_end_max,
                                               'box_id': label, 'grouped': 1,
                                               'confidence': mean_confidence}]
                                         )
                                         ],
                                        ignore_index=True)
            else:
                # Include boxes that did not overlap in the final DataFrame
                boxes_agrup = pd.concat([boxes_agrup, filtered.assign(grouped=0)], ignore_index=True)

    return boxes_agrup
