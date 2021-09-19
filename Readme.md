Source code for https://www.linkedin.com/feed/update/urn:li:activity:6844867100176191488/

install windows:
conda install -c intel openvino-ie
pip install labelme2coco
pip install openvino
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install ./car/lib/openvino_vehicle_detection_adas-0.2.1+cpu-py3-none-any.whl
pip install ./car/lib/openvino_vehicle_license_plate_detection_barrier-0.2.1+cpu-py3-none-any.whl

install linux:
pip install openvino
pip install labelme2coco
pip install openvino
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install ./car/lib/openvino_vehicle_detection_adas-0.2.1+cpu-py3-none-any.whl
pip install ./car/lib/openvino_vehicle_license_plate_detection_barrier-0.2.1+cpu-py3-none-any.whl
