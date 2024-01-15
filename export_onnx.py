
import argparse
import os
import sys
import platform
from pathlib import Path

import torch
from brevitas.export import export_brevitas_onnx



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str, default='experiment_models/lpyolo_W4A4.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='models/lpyolo_quant.yaml', help='model config')
    parser.add_argument('--classes', type=int, default=7, help='number of classes')
    parser.add_argument('--output_dir',type=str,default='experiment_models')
    opt = parser.parse_args()

    return opt


def main(opt):
    model = torch.hub.load(
        '.',
        'custom',
        str(opt.weights),
        source='local',
        classes = 7,
        force_reload=True,
        cfg = str(opt.cfg),
    )


    IN_CH = 384
    OUT_CH = 640
    BATCH_SIZE = 1

    model_no_detect = torch.nn.Sequential(*[model.model.model.model[i] for i in range(15)])

    model_name = opt.weights.split("/")[-1].replace(".pt","")

    path = f'{opt.output_dir}/{model_name}.onnx'
    inp = torch.randn(BATCH_SIZE,3, IN_CH, OUT_CH).cuda()

    detection_model = model_no_detect
    detection_model.cuda()
    detection_model.eval()

    exported_model = export_brevitas_onnx(detection_model, inp, path)

    detect_module = model.model.model.model[15]
    torch.save(detect_module.state_dict(), f'{opt.output_dir}/detect_module.pt')

    print(f"saving complete to {opt.output_dir}")


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
