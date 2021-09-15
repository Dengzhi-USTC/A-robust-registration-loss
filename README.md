# [A-robust-registration-loss](https://arxiv.org/pdf/2108.11682.pdf)
Deep learning, rigid registration, intersected lines, unsupervised learning, uniform distributions, iccv2021.
![](./data/introduce_our_loss.png)
## Setup
### Requirements
  [Our metric](https://arxiv.org/abs/2108.11682) are implemented with the Pytorch, we test on the Pytorch[0.7, 1.7] are available, we run our metric on the V100 and 3090, considering memory consumption, please keep the memory above 15G. More details, refer the [requirements.txt](./code/requirements.txt).
### Data
  
  You can download the [Human dataset](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhideng_mail_ustc_edu_cn/EZ1nYTksRa1JndRj7c6wV4IB9wfSr3ataJV8NE0b4EZYtQ?e=PIJsFB), [Airplane datasets](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhideng_mail_ustc_edu_cn/EflslRBzK6pBmBtcaWoU8lsBnUSvm74JIG99Et9Rxo8xqQ?e=AjkdSU), [Real dataset](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhideng_mail_ustc_edu_cn/EW8GRJG9cGRLjI0qnED90o8BJ-zTWjp9B_Y3TT4tQPncEQ?e=d2RASg).

  You also can use our [scripts](./code/generate_data_preparation.py) to generate neigh-points to retrain on your datasets, but please refer to the training details in our papers.

### Metric

  Our metrics have different versions from the beginning of the discussion. The overall research route is from mesh data to general point cloud data. The specific loss form can refer to [loss.py](./code). Looking forward to extending our measurement to other frameworks or other areas.
- Experiments
  - Optimization of a single example by embedding the metric into the traditional optimization based on Adam [Demo](./code)
  ![](./data/supp_real_exp-1.png)
  - Embed our metrics into deep learning and transform supervised frameworks into unsupervised frameworks, ([RMP-Net](./experiments), [DCP](./experiments), [FMR](./experiments)). [Exp-DL](./code/exps_deep_learning) We also provided the [pretrained models](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhideng_mail_ustc_edu_cn/EZ1nYTksRa1JndRj7c6wV4IB9wfSr3ataJV8NE0b4EZYtQ?e=PIJsFB),
- Cost computation
  
![](./data/Computation_cost.png)

- Visualize the the energy of optimizing a single example(Refer the [More visualization](./More_about_our_metrics/Visualized_our_metrics.md))


### BibTex
    @article{dengzhi2021robust, 
    title={A Robust Loss for Point Cloud Registration}, 
        author={Zhi Deng, Yuxin Yao, Bailin Deng and Juyong Zhang},
        journal={ICCV},
    year={2021} 
}
### Notes
If you have comments or questions, please contact Zhi Deng([zhideng@mail.ustc.edu.cn]()), Yao Yuxin ([yaoyuxin@mail.ustc.edu.cn]()), Bailin Deng ([DengB3@cardiff.ac.uk]()), Juyong Zhang ([Juyong Zhang@ustc.edu.cn]()).
### Acknowledgement

  We would like to thank the authors of [DCP_code](https://github.com/tzodge/PCR-CMU/tree/main/DCP_Code), [RPM-Net_code](https://github.com/tzodge/PCR-CMU/tree/main/RPMNet_Code), [FMR_code](https://github.com/XiaoshuiHuang/fmr), [FRICP](https://github.com/yaoyx689/Fast-Robust-ICP), [FGR](https://github.com/isl-org/FastGlobalRegistration) for making their codes available.

  We can certainly thank the source of the data set, [Human dataset](https://secure.axyz-design.com//), [M40](https://github.com/zhirongw/3DShapeNets), Partial Real-datasets, [3D-Match](https://arxiv.org/pdf/1603.08182.pdf), [7scenes](https://openaccess.thecvf.com/content_cvpr_2013/papers/Shotton_Scene_Coordinate_Regression_2013_CVPR_paper.pdf), [SLAM](https://www.researchgate.net/publication/261353760_A_benchmark_for_the_evaluation_of_RGB-D_SLAM_systems).


