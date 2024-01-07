# Quant LPYOLO

The configuration files of the lpyolo and other versions of yolo can be found under `models/` 

The weight and activation quantization bitwidth can be modified in `models/lpyolo_quant.yaml`

Dataset configuration can be found under `data/`
The two relevant datasets to us are `coco128` and `VOC`


## Train

Training is done on coco128 dataset with these classes
0: person
1: bicycle
2: car
3: motorcycle
4: bus
5: train
6: truck

### Train Unquantized

```sh
python3 train.py --img 640 --batch 64 --epochs 300 --data coco128.yaml --weights '' --cache --cfg models/lpyolo.yaml --classes 7
```

### Finetune quantized model (QAT)
```sh
!python3 train.py --img 640 --batch 32 --epochs 50 --data coco128.yaml --weights /path/to/lpyolo.pt --cache --cfg models/lpyolo_quant.yaml --classes 7
```

## Val

Validation can either be done on coco128 or VOC datasets

```sh
python3 val.py --weights /path/to/lpyolo.pt --data VOC.yaml --img 640 --half
```

## Detect

### Detect with unquantized model
```sh
python3 detect.py --cfg models/lpyolo.yaml --weights /path/to/lpyolo.pt --img 640 --conf 0.25 --source /path/to/image
```
### Detect with Quantized model
```sh
python3 detect.py --cfg models/lpyolo_quant.yaml --weights /path/to/lpyolo_quant.pt --img 640 --conf 0.25 --source /path/to/image
```

## Saving the Model

```sh
python3 export.py --cfg models/lpyolo_quant.yaml --weights /path/to/lpyolo_quant.pt --classes 7 --output_path /to/output_path
```

The output will be the onnx model and the detect module pt file
