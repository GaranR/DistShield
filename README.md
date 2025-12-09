# "DISTSHIELD: Distribution Preserving Model Obfuscation for Real-Time TEE-Shielded Secure Inference on IoT devices"

The code for implementing the ***DISTSHIELD: Distribution Preserving Model Obfuscation for Real-Time TEE-Shielded Secure Inference on IoT devices***.

## Requirements

### Python Environment

Our code is tested in python 3.6. We provide a simple instruction to configure the essential python libraries.

```
pip install -r requirements.txt
```

### **Intel SGX Hardware** and **Gramine**

A device equipped with Intel SGX is required as the hardware to run the `TEE_test` part. It is recommended to test on Linux since we have not tested Gramine in Windows.

The following steps are necessary to build a Gramine environment.

1. Linux-SGX Driver. SGX-Driver is required to be installed, which is the fundemental environment. Please refer to [Linux-SGX Respository](https://github.com/intel/linux-sgx) to build from source-code. For some versions of CPUs and systems, SGX may already be integrated in the system driver.

2. Gramine-SGX. Gramine-SGX is a libOS which supports runing application in SGX without modification. Please follow the [Gramine Respository](https://github.com/gramineproject/gramine) to install the Gramine.

3. Test. You can test your Gramine according to this simple [Demo](https://github.com/gramineproject/examples/tree/master/pytorch).

### Dataset

You are supposed to download [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [STL-10](https://cs.stanford.edu/~acoates/stl10/), [RESISC-45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html), [LFW](https://complexity.cecs.ucf.edu/lfw-labeled-faces-in-the-wild/) or any dataset you want, and modify the dataset path and transform parameters in the code.

### Models

Prepare the pre-trained model you want to process with DistShield and change the model import path.

### Folder Structure:

```
Distshield
   ├── attack/                [The model stealing attack in Section 3 of the paper.]
   ├── distshield/            [The DistShield for models using CE-LOSS.]
   ├── distshield_facenet/    [The DistShield for FaceNet, modified for triplet loss.]
   ├── distshield_LLMs/          [The DistShield for LLMs.]
   ├── TEE_test/              [Testing the TEE computation latency, requires SGX hardware.]
   └── requirements.txt       [The necessary dependencies for the project]
```

## Running Experiments

### Model Stealing Attack

Modify the paths for the pre-trained model and the obfuscated model in `attack.py`, then simply run:

```
cd attack && python attack.py
```

### DistShield

To generate an obfuscated model from a pre-trained model using ***DistShield***, you need to specify the pre-trained model by setting `net` in `distshield.py`, and `net_ori` and `net_tmp` in `train.py`. Then run:

```
cd distshield && python distshield.py --model 0 --dataset 0 --lr 0.005 --num_epoch 20 --PATH ./result/my_temp.pth --random 0 --opti 1 --ratio 0.001
```

You can specify the storage path for the obfuscated model using `--PATH`and control the initial number of obfuscated parameters before Iterative Weight Pruning using `--ratio`.

The `random` option enables random parameter selection, while the `opti` option toggles Iterative Weight Pruning. These two parameters are used for experiments in ablation study.

`magnitude_text.py` is used to generate a model obfuscated using the ***Magnitude***. Simply set the original model and run:

```
cd distshield && python magnitude_test.py
```

### DistShield for FaceNet

First place your training dataset in `/distshield_facenet/datasets` and the LFW dataset in `/distshield_facenet/lfw`, and run:

```
cd distshield_facenet && python txt_annotation.py
```

Then change `model_path` in `distshield_facenet.py` to the path of your pre-trained FaceNet backbone, and run:

```
cd distshield_facenet && python distshield_facenet.py
```

The ***DistShield*** obfuscated model is saved by default as `distshield_facenet.pth`. You can change this default path in `distshield_facenet/utils/utils_fit.py`.

You can set `train = 0` and `test_mag = 1` in `distshield_facenet.py` to generate the obfuscated model using the ***Magnitude***.

### DistShield for LLMs

This folder provides demonstrations of ***DistShield*** applied to large language models (LLMs), covering three tasks: BERT(MRPC), RoBERTa(SQuAD), and Qwen-3B(GSM8K).

To run the experiments, simply install Jupyter Notebook and open the corresponding `.ipynb` files. Each notebook will walk you through the ***DistShield*** model obfuscation process and display the results automatically—no additional setup is needed.

### Test Computation Latency in TEE

To test the computation latency in TEE, make sure that Intel SGX Hardware and Gramine are properly configured first. You may need to configure the `pytorch.manifest.template` according to your own Gramine settings

You need to specify the obfuscated model and the original model for testing in `gen_input.py` and `test_time.py`. The obfuscated models for ***DistShield*** and ***Magnitude*** are generated by our code, while the obfuscated model for ***NNSplitter*** is generated using its [open-source code](https://github.com/Tongzhou0101/NNSplitter). The ***GroupCover*** scheme differs significantly, and we generate its model key and test the speed based on its [open-source code](https://github.com/ZzzzMe/GroupCover) in `test_time.py`.

First, run:

```
cd TEE_test && python gen_input.py
```

to save the input of each layer of the original model in the `/TEE_test/layers` folder.

Then, test the computation latency in TEE using Gramine:

```
cd TEE_test && gramine-sgx ./pytorch test_time.py
```

## License

This project well be released under the MIT License if accepted.