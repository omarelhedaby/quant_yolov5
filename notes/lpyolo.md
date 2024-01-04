## Issues
- **1** Input of the model is between 0 .. 1 while on finn the input is uint8 (0 .. 255), perhaps that is why the output is wrong on finn? But also I tried the whole model including detect with ModelWrapper and the output was fine, so perhaps that is not the problem.

- **2** when I input a normal pic to the model,it does not work. 
FIXED (preprocessing)

- **3** How does brevitas do quantization in regards to this paper [A Survey of Quantization Methods for Efficient
Neural Network Inference](https://www.semanticscholar.org/reader/04e283adccf66742130bde4a4dedcda8f549dd7e)

- **4** Choice of iou threshold

- **5** which optimizer leads best results? SGD with lr = 0.01

## Findings
- **1**
  - I tried the full model with ModelWrapper and it works only with float input.
  - Also that shows that the input quantization type does not really matter.
  - I am still confused about how finn takes the input as UINT8, am I supposed to 
  dequantize it with the scale which I can get from the Netron of the cleaned model?
  - Does the divided model gets correct output using python? yes
  - ![Alt text](multithreshold_in_quant.png)
  - ![Alt text](threshold_batch_in_quant.png)
  - we can see that the scales were turned to 0s and 1s, strange.
  - so now I will train a model without input quantization and check what happens
  - I will also train a model with UINT8 input quantization
  - we could also train the image with images with 0..255, (Solution)
  - input normalization was removed at val.py, train.py and common.py, it will be found after 
  #TODO change according to accelerator

- **2**
  - the input appears to be preprocessed where 9 pics are added on top of each other and they are all grayscale. 
  - also the size of the preprocessed images is 640,480 not 384. that could be the problem.

- **3**
  - Default quantization method Int8WeightPerTensorFloat is fake quantization or simulated quantization as it is mentioned in the paper. ![Alt text](simulated_quantization.png)
  This method is slower than Integer-only quantization because it does not benefit from low precision logic, however, it can prove to have better acurracy and is more suitable for applications like recommendation systems.however that is on Brevitas, I believe FINN just uses the int weights for quantization as in the graphs on Netron. perhaps Fixed Weight is best for training. 

  - Both Fake Simulated quantization and Integer Only are limited to RELU activation.
  - Int8WeightPerTensorFixedPoint limits the scale to a power of 2.
  - SignedBinaryWeightPerTensorConst is used for binary quantization
  - **Note** how by default the output tensor is returned as a standard torch tensor in dequantized format. This is designed to minimize friction with standard torch operators in the default case. To return an output QuantTensor we have to set return_quant_tensor=True
  - **QuantIdentity** layer. By default QuantIdentity adopts the Int8ActPerTensorFloat quantizer
  - **QuantReLU** the default quantizer is Uint8ActPerTensorFloat. Quantized activations layers like QuantReLU by defaults quantize the output, meaning that a relu is computed first, and then the output is quantized
  - When using a Quantized Detect module, I would get an error because of bias and input scale, so the solution is either passing a Quant Tensor which in our case is not possible because the input is the output of the accelator which is a normal Tensor, so the solution would be an input quantizer Int8ActPerTensorFloat to fix this issue.

  - **4** when training use a high threshold like 0.6 to remove only highly overlapping boxes, leaving detections with lower IoU for the model to learn how to precise the bounding box location better.

  while inference, a lower threshold is more likely to be used to filter overlapping boxes for the same detection with different anchor box sizes for example. 0.4 is a good iou.

  as the way iou is done is sorting accoriding to confidence and assign the predictions with highest scores and that have no high overlapping with each other as detections. and then doing NMS to filter the bounding boxes that have overlapping with these detections, in case of inference, use a lower threshold to avoid having overlapping detections in the prediction.

  also the threshold cant be too low, this can result in filtering valid detection with lower confidence than another detection that is overlapping with. in the case of a crowded road for example you can except to have many overlapping pedestrians and cars with different confidences, so a very low threshold can discard valid detections