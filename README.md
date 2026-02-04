# RIVA-challenge

<p align="center">
    <a href="https://www.kaggle.com/competitions/riva-cervical-cytology-challenge-isbi-2026/overview">
        <img src="assets/banner.png" alt="RIVA Challenge" width="800"/>
    </a>
</p>

This repository is our solution to the [RIVA Cervical Cytology Challenge](https://www.kaggle.com/competitions/riva-cervical-cytology-challenge-isbi-2026/overview) Track A in the context of ISBI 2026.

This project has the objective of finding the best architecture in order to both detect and classify cells of the Bethesda categories (NILM, ASCUS, LSIL, HSIL, ASC-H, SCC, INFL, ENDO).

The first model architecture that we proposed consisted of using [SAM3](https://arxiv.org/pdf/2511.16719)'s vision encoder as a backbone of [Faster-RCNN](https://arxiv.org/pdf/1506.01497). It  is a combination of a SOTA model (SAM3) and a well established one (Faster-RCNN).

The second model architecture, and the current one, also uses SAM3's vision encoder but this time as a backbone of [DETR](https://arxiv.org/pdf/2005.12872). This version of the model is much newer and takes advantage of the current Transfomer based architectures. 

The third model architecture, and the one we will begin experimenting with, uses [Cell-DINO](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013828) as a backbone of Faster-RCNN. This model will potentially profit from the pretraining of the backbone on cell images, leading to a stronger performance on the RIVA dataset.

## 🚀 Quick start
Start by downloading the projects dependencies by running
```cli
pip install -r requirements.txt
```
>Added one more dependency, don't forget to install it to prevent ImportError!

To access the challenges training, validating and testing datasets with their respective annotations simply run ensuring you are already participating in the challenge.
```cli
kaggle competitions download -c riva-cervical-cytology-challenge-isbi-2026
```

## 🦾 Training
First choose which model you want to train from the ones that are in "models/" and then run:
```cli
python train.py --model <MODEL_NAME>
```
The only valid model names are: 
- **sam3_rcnn** for the SAM3 + Faster-RCNN 
- **sam3_detr** for the SAM3 + DETR
- **sam3_rcnn_v2** for the *revisited* SAM3 + Faster-RCNN

We also have two other specific training loops: one for the *revisited* SAM3 + DETR and one for the Cell-DINO + Faster-RCNN model.

To train the revisited SAM3 + DETR run:
```cli
python train_sam3_detr_v2.py
```
For tuning the hyperparameters, the training script takes the following arguments:
- `--freeze_sam3`: If set, freeze the SAM3 backbone (only train the classification head)
- `--checkpoint_dir`: Directory to save checkpoints
- `--resume`: Path to checkpoint to resume training from
- `--batch_size`: Batch size for training (default: 8, adjust based on GPU memory)
- `--epochs`: Number of training epochs
- `--lr_backbone`: Learning rate for SAM3 backbone
- `--lr_head`: Learning rate for classification head
- `--score_thresh`: Score threshold for inference during validation
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)

Before training the Cell-DINO model, you must create a .env on the root directory with the following format:
```txt
CELL_DINO_WEIGHTS_URL=<WEIGHTS_URL>
DEBUG=TRUE
```
> Ensure you have python-dotenv installed!

To obtain the weights url complete the requested information in the following link: https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/.
After that, you should receive an email with the url's to the weights. The one to use is
**cell_dino_vitl14_pretrain_hpa_fov_highres-*.pth**

The training script for Cell-DINO + Faster-RCNN is much simpler. It can be run with:
```cli
python train_cell_dino.py
```
And it only has the following arguments:
- `--pretrained_checkpoint_path`: Path to Cell-DINO high-res weights (.pth file). Optional if loading from URL.
- `--trainable_backbone`: If set, unfreeze backbone (fine-tuning). Default is Frozen.

## 📖 Testing
To run the functionality tests run:
```
python tests/
```

## 📁 Generating predictions

To generate predictions over the test dataset run:
```cli
python predict.py --model <MODEL_NAME>
```
This will generate a submission.csv file in the "results/" directory.

##  📜 References

If you use this code in your research, please cite:

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts}, 
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719}, 
}

@misc{ren2016fasterrcnnrealtimeobject,
      title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks}, 
      author={Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
      year={2016},
      eprint={1506.01497},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1506.01497}, 
}

@misc{carion2020endtoendobjectdetectiontransformers,
      title={End-to-End Object Detection with Transformers}, 
      author={Nicolas Carion and Francisco Massa and Gabriel Synnaeve and Nicolas Usunier and Alexander Kirillov and Sergey Zagoruyko},
      year={2020},
      eprint={2005.12872},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2005.12872}, 
}

@misc{,
  title={Cell-DINO: Self-Supervised Image-based Embeddings for Cell Fluorescent Microscopy},
  author={Moutakanni, Th\'eo and Couprie, Camille and Yi, Seungeun and Gardes, Elouan Gardes and Bojanowski, Piotr and Touvron, Hugo and Doron, Michael and Chen, Zitong S. and Moshkov, Nikita and Caron, Mathilde and Joulin, Armand and Pernice, Wolfgang M. and Caicedo, Juan C.},
  journal={in review to PloS One on Computational Biology},
  year={2025}
}
```