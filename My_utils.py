import json
import csv
import os
import re

def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(q.numel() for q in model.buffers())
    print("\033[1;32;m{}\033[0m model have \033[1;32;m{}\033[0m parameters.".format(model.__class__.__name__, total_params + total_buffers))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\033[1;32;m{}\033[0m model have \033[1;32;m{}\033[0m training parameters.".format(model.__class__.__name__, total_trainable_params))

def json2csv(json_name, class_struct_level=1):
    with open(json_name, "r") as f:
        meta = json.load(f) # json file(dict) ： {"label_names:[...], "image_names":[...], "image_labels":[...]}

    image_names = meta["image_names"]
    image_labels = meta["image_labels"]
    label_names = meta["label_names"]
    csv_name = json_name.replace(".json", ".csv")
    assert len(image_names) == len(image_names), "number of names must match number of labels"
    head = ["filename", "label"]
    with open(csv_name, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(head)

        for i in range(len(image_names)):
            if class_struct_level == 1:
                writer.writerow([os.path.basename(image_names[i]), label_names[image_labels[i]]])
            elif class_struct_level == 2:
                writer.writerow(["/".join(image_names[i].split("/")[-2:]), label_names[image_labels[i]]])
        
        print("{} write complete".format(csv_name))

def args2launch(string):
    config = re.sub("(?<=\s)([\S.]*)(?=\s)", "\"\\1\",\n", string, count=0).strip()
    return config


if __name__ == "__main__":
    # json_name = "dataset/VHR-10/novel.json"
    # json2csv(json_name, 2)
    # print()
    config = args2launch(" --resume ./results/DN4_miniImageNet_Conv64F_5Way_1Shot_K3/model_best.pth.tar --basemodel Conv64F ")
    print(config)