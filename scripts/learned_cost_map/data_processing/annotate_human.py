import csv
import os
import yaml
fp = "/home/mateo/Data/SARA/TartanCost/human_labels.csv"
annotations_dir = "/home/mateo/Data/SARA/TartanCost/Annotations"
with open(fp) as csvfile:
    reader = csv.reader(csvfile)
    fields = next(reader)
    print(f"Fields are: {fields}")
    for i,row in enumerate(reader):
        print("-----")
        print(f"Row {i}")
        print(row)

        annotation_fp = os.path.join(annotations_dir, f"{i:06}.yaml")
        print(f"Annotation filepath is: {annotation_fp}")

        with open(annotation_fp, 'r') as anno_file:
            annotation = yaml.safe_load(anno_file)
        print(annotation)

        human_scores = {
            "average": float(row[4]),
            "score_1": float(row[1]),
            "score_2": float(row[2]),
            "score_3": float(row[3])
        }

        print("Human scores: ")
        print(human_scores)

        annotation["human_scores"] = human_scores

        with open(annotation_fp, 'w') as outfile:
            yaml.safe_dump(annotation, outfile, default_flow_style=False)