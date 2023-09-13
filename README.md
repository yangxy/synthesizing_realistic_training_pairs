# Synthesizing Realistic Image Restoration Training Pairs: A Diffusion Approach

[Paper](https://arxiv.org/abs/2303.06994)

[Tao Yang](https://cg.cs.tsinghua.edu.cn/people/~tyang)<sup>1</sup>, Peiran Ren<sup>1</sup>, Xuansong Xie<sup>1</sup>, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>2</sup>  
_<sup>1</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com), Hangzhou, China_  
_<sup>2</sup>[Department of Computing, The Hong Kong Polytechnic University](http://www.comp.polyu.edu.hk), Hong Kong, China_

## Real-ISR
<img src="samples/0014.gif" width="390px"/> <img src="samples/dped_crop00061.gif" width="390px"/>
<img src="samples/00003.gif" width="390px"/> <img src="samples/00017_gray.gif" width="390px"/>

## News
(2023-09-13) Upload pre-trained models.
(2023-09-07) Upload source codes.

## Usage
- Clone this repository:
```bash
git clone https://github.com/yangxy/synthesizing_realistic_training_pairs.git
cd synthesizing_realistic_training_pairs
```

- Prepare LQ/HQ datasets, e.g., [DID_natural](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SR_models/did_natural_urls.txt)/[DF2K_OST](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost), and put them into ``datasets/``. Please send me email for DID_face dataset.

- Train a DDPM that generates realistic LQ images.
```bash
bash ./train_ddpm.sh
```

Download our pre-trained model [ddpm_did_256](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SR_models/checkpoint-50000.zip) trained with a resolution of ``256x256``. 

- (Optional) Extrat HQ images to subimages in order to match your train size.
```bash
python scripts/extract_subimages.py # change the values of opt, especially opt['input_folder']/opt['save_folder'] accordingly
```

- Synthesize realistic LQ images with the help of the pre-trained DDPM.
```bash
python test_ddpm_img2img.py --max_strength 0.2
```

- Train your own SR models using the synthesized HQ-LQ pairs

- Test your SR model.

You can download our pre-trained models [RRDB+](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SR_models/RRDB%2B.pth). 

#### We re-train our models due to the whole project is re-builed on [diffusers](https://github.com/huggingface/diffusers). The outputs may differ from the results presented in the paper. We are still working on it and the released models woule be updated at any time. 

## Citation
If our work is useful for your research, please consider citing:

    @inproceedings{yang2023syn,
	    title={Synthesizing Realistic Image Restoration Training Pairs: A Diffusion Approach},
	    author={Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang},
	    booktitle={Arxiv},
	    year={2023}
    }
    
## License
© Alibaba, 2023. For academic and non-commercial use only.

## Acknowledgments
Our project is based on [diffusers](https://github.com/huggingface/diffusers)

## Contact
If you have any questions or suggestions about this paper, feel free to reach me at yangtao9009@gmail.com.
