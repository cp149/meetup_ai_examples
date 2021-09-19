import sys
import os
from PIL import Image, ImageDraw
from openvino_vehicle_license_plate_detection_barrier import (
    InferenceModel,
)  # import the AI Model
from modelplace_api import Device

base_dir = "./car"
# pip install ./car/lib/openvino_vehicle_license_plate_detection_barrier-0.2.1+cpu-py3-none-any.whl


def run_model(path_to_image: str):
    image = Image.open(path_to_image).convert("RGB")  # Read an image
    model = InferenceModel()  # Initialize a model
    model.model_load(Device.cpu)  # Loading a model weights
    ret = model.process_sample(image)  # Processing an image
    return ret


def run():
    for test_file in [
        x for x in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, x))
    ]:
        test_file = os.path.join(base_dir, test_file)
        out_put = run_model(test_file)
        print(out_put)
        source_img = Image.open(test_file).convert("RGB")
        draw = ImageDraw.Draw(source_img)
        for item in out_put:
            draw.rectangle(
                ((item.x1, item.y1), (item.x2, item.y2)),
                fill=None,
                outline="red",
            )
        source_img.show()


if __name__ == "__main__":
    run()
    # pip install ./car/lib/openvino_vehicle_detection_adas-0.2.1+cpu-py3-none-any.whl
    from openvino_vehicle_detection_adas import (
        InferenceModel,
    )  # import next the AI Model

    run()
