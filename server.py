#!/usr/bin/env python3

import pickle

from pathlib import Path
from uuid import uuid4

from flask import Flask, request, render_template

import cv2
import numpy as np

from query import images_similar_to
from faster_rcnn import Config
from create_retrieval_db import best_bbox

app = Flask(__name__, static_url_path="", static_folder="dist")

app.config["ENV"] = "development"

q_path = Path("dist/query")
res_path = Path("dist/results")

styles = []


@app.route("/search", methods=["POST"])
def search():
    files = request.files
    req_uid = str(uuid4())
    up_name = (q_path / req_uid).with_suffix(".jpg")
    if not "file" in files:
        # TODO: handle error
        print(files)
        return

    file = files["file"]
    file.save(str(up_name))
    sim_images, result = images_similar_to(
        str(up_name), features_per_class, metadata_per_class, C
    )
    if not str(up_name) in result:
        return render_template(
            "result.html", qimg=str(up_name.relative_to("dist")), imgs=[]
        )

    instance = result[str(up_name)]
    best_is = best_bbox(instance, n=None)
    img = cv2.imread(str(up_name))

    classes_found = {i["class"] for i in sim_images}
    legends = [
        {
            "class_color": "rgb("
            + str(class_to_color[cl][0])
            + ","
            + str(class_to_color[cl][1])
            + ","
            + str(class_to_color[cl][2])
            + ")",
            "class_name": " ".join(cl.split("_")),
        }
        for cl in classes_found
    ]
    for best_i in best_is:
        claz = instance[1][best_i][1]

        (x1, y1, x2, y2) = instance[0][best_i]

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (
                int(class_to_color[claz][2]),
                int(class_to_color[claz][1]),
                int(class_to_color[claz][0]),
            ),
            4,
        )
    cv2.imwrite(str(up_name), img)

    print(legends)
    print(list(sim_images)[:15])
    return render_template(
        "result.html",
        qimg=str(up_name.relative_to("dist")),
        imgs=list(sim_images)[:15],
        styles=styles,
        legends=legends,
    )


if __name__ == "__main__":
    with open("data/instre_monuments/model_vgg_config.pickle", "rb") as f_in:
        C = pickle.load(f_in)
        # Switch key value for class mapping

    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    class_to_color = {
        class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping
    }

    styles = [{"name": c, "color": class_to_color[c]} for c in class_to_color]

    with open("retrieval_db/features_per_class", "rb") as f:
        features_per_class = pickle.load(f)

    with open("retrieval_db/metadata_per_class", "rb") as f:
        metadata_per_class = pickle.load(f)

    app.run()
