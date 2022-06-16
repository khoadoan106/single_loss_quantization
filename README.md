# Implementations of Deep Hashing Models

## Prerequisites
We need the following:
* conda or miniconda (preferred)
* GPU or CPU

## Setup the environment
Clone the repository. The setup script to initialize and activate the environment is collected in `etc/setup_env`. Simply run the following command:
```
. etc/setup_env
```
## Repository artifacts

* `python`: code folder
* `requirements.txt`: list of python reqs
* `README.md`: this doc, and light documentation of this repos.

## Using Original Deep Hashing Methods
* CIFAR10:
        ```
        nohup python python/HashNet.py --dataset cifar10 --data_root data/ --random_seed 9 --bit_list 16 32 64 128 --model AlexNet --epochs 200 --optimizer adam --lr 1e-5 --test_every 10  --save_path experiments/HashNet/cifar10_AlexNet_b64_adam  2>&1 >experiments/logs/HashNet-r9-cifar10_AlexNet_b64_adam.log &	
        ```
* COCO:
        ```
        nohup python python/HashNet.py --dataset coco --data_root data/ --random_seed 9 --bit_list 16 32 64 128 --model AlexNet --epochs 200 --optimizer adam --lr 1e-5 --test_every 10  --save_path experiments/HashNet/coco_AlexNet_b64_adam  2>&1 >experiments/logs/HashNet-r9-coco_AlexNet_b64_adam.log &	
        ```
* NUS-WIDE:
        ```
        nohup python python/HashNet.py --dataset coco --data_root data/ --random_seed 9 --bit_list 16 32 64 128 --model AlexNet --epochs 200 --optimizer adam --lr 1e-5 --test_every 10  --save_path experiments/HashNet/nuswide_21_AlexNet_b64_adam  2>&1 >experiments/logs/HashNet-r9-nuswide_21_AlexNet_b64_adam.log &	
        ```   

We currently support the following methods
* Cao et al. **HashNet: Deep Learning to Hash by Continuation**. ICCV 2017. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)] [[HashNet.py](python/HashNet.py)]
* Li et al. **Deep Supervised Discrete Hashing**. NIPS 2017. [[Paper](https://proceedings.neurips.cc/paper/2017/file/e94f63f579e05cb49c05c2d050ead9c0-Paper.pdf)] [[DSDH.py](python/DSDH.py)]
        
## Using Distributional Quantization Approach
Doan et al. One Loss for Quantization: Deep Hashing with Discrete Wasserstein Distributional Matching (CVPR2022). [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Doan_One_Loss_for_Quantization_Deep_Hashing_With_Discrete_Wasserstein_Distributional_CVPR_2022_paper.pdf)]

* This repository supports various quantization losses discussed in Doan et al. For HSWD, use `--quantization_type swdC`; for SWD, use `--quantization_type swd`; we also support Optimal Transport estimation using `--quantization_type ot`.
    
* CIFAR10:
        ```
        nohup python python/HashNet.py --dataset cifar10 --data_root data/ --random_seed 9 --bit_list 16 32 64 128 --model AlexNet --epochs 200 --optimizer adam --lr 1e-5 
        --quantization_type swdC --quantization_alpha 0.1 --test_every 10  --save_path experiments/HashNet/cifar10_AlexNet_b64_adam  2>&1 >experiments/logs/HashNet-r9-cifar10_AlexNet_b64_adam_swdC.log &	
        ```
* COCO:
        ```
        nohup python python/HashNet.py --dataset coco --data_root data/ --random_seed 9 --bit_list 16 32 64 128 --model AlexNet --epochs 200 --optimizer adam --lr 1e-5 --quantization_type swdC --quantization_alpha 0.1 --test_every 10  --save_path experiments/HashNet/coco_AlexNet_b64_adam  2>&1 >experiments/logs/HashNet-r9-coco_AlexNet_b64_adam_swdC.log &	
        ```
* NUS-WIDE:
        ```
        nohup python python/HashNet.py --dataset coco --data_root data/ --random_seed 9 --bit_list 16 32 64 128 --model AlexNet --epochs 200 --optimizer adam --lr 1e-5  --quantization_type swdC --quantization_alpha 0.1 --test_every 10  --save_path experiments/HashNet/nuswide_21_AlexNet_b64_adam  2>&1 >experiments/logs/HashNet-r9-nuswide_21_AlexNet_b64_adam_swdC.log &	
        ```        
        
## Citations
Please cite the following work when using this repository:

```
@inproceedings{doan2022one,
  title={One Loss for Quantization: Deep Hashing with Discrete Wasserstein Distributional Matching},
  author={Doan, Khoa D and Yang, Peng and Li, Ping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9447--9457},
  year={2022}
}
```

## Acknowledgements
* This respository is inspired by this respository https://github.com/swuxyj/DeepHash-pytorch. Thank you the authors of DeepHash-pytorch.
