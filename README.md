# RIVA-challenge

<p align="center">
    <a href="https://www.kaggle.com/competitions/riva-cervical-cytology-challenge-isbi-2026/overview">
        <img src="assets/banner.png" alt="RIVA Challenge" width="800"/>
    </a>
</p>

This repository is our solution to the [RIVA Cervical Cytology Challenge](https://www.kaggle.com/competitions/riva-cervical-cytology-challenge-isbi-2026/overview) Track A in the context of ISBI 2026.

This project has the objective of finding the best architecture in order to both detect and classify cells of the Bethesda categories (NILM, ASCUS, LSIL, HSIL, ASC-H, SCC, INFL, ENDO).

The current model architecture that we propose consists of using [SAM3](https://arxiv.org/pdf/2511.16719)'s vision encoder as a backbone to [Faster-RCNN](https://arxiv.org/pdf/1506.01497). Our proposed solution is a combination of a SOTA model (SAM3) and a well established one (Faster-RCNN).

## 🚀 Quick start
Start by downloading the projects dependencies by running
```cli
pip install -r requirements.txt
```

To access the challenges training, validating and testing datasets with their respective annotations simply run
```cli
kaggle competitions download -c riva-cervical-cytology-challenge-isbi-2026
```

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
```