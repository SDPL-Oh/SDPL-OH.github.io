---
layout: post
title:  "MMDETECTION 실행 파일"
date:   2022-07-14 09:00:00 +0530
categories: mmdetection 
---
mmdetection 실행 방법에 대해 정리한다.  


<br/>

## Custom config
config 파일은 대부분 네트워크가 구축되어있고, 기본 config 파일을 통해 가지치기 하고 있다.
사용자화하려면 네트워크, 학습 옵션을 별도로 작성해서 실행해야한다.

```
mmdetection>configs>_base_>datasets
mmdetection>configs>_base_>models
mmdetection>configs>_base_>schedules
```

<br/>

## Dataset
사용자의 데이터셋을 추가하기 위해 데이터 config 파일을 작성하고 등록해야한다.

```
mmdetection>mmdet>datasets>coco.py
```

../mmdet/datasets/에 파일 생성하고, @DATASETS.register_module(), __init__파일에서 import와 __all__에 dataset 추가해야한다.


<br/>

## Operator
실행할 수 있는 버전은 두 가지가 있다.
윈도우 버전에서는 train.py를 사용하며, 리눅스에서는 dirt_train.sh를 사용한다. sh를 사용할 경우 multi-gpu 설정이 가능하다.
```
mmdetection/tools/train.py
mmdetection/tools/dist_train.sh
```
### train.py
```
python mmdetection/tools/train.py /
mmdetection/configs/custom/faster_rcnn_r50.py
```
### dirt_train.sh
```
mmdetection/tools/dist_train.sh /
mmdetection/configs/custom/faster_rcnn_r50.py /
4
```

<br/>

## Evaluation
mmdetection에는 학습 결과를 평가하고, 검토할 수 있는 기능이 갖춰져 있다. 학습과정에서 생성된 로그 파일이 있어야하며, test.py을 통해 pkl, bbox.json이 생성하는 과정을 거쳐야한다.
```
mmdetectiontools/analysis_tools/analyze_logs.py
mmdetection/tools/analysis_tools/analyze_results.py
mmdetection/tools/analysis_tools/confusion_matrix.py
mmdetectiontools/analysis_tools/coco_error_analysis.py
```

### pkl 생성
```
python mmdetection/tools/test.py /
mmdetection/configs/custom/faster_rcnn_r50.py /
work_dirs/faster_rcnn_r50/latest.pth /
--eval bbox /
--out results.pkl
```
### bbox.json 생성
```
python mmdetection/tools/test.py /
mmdetection/configs/custom/faster_rcnn_r50.py /
work_dirs/faster_rcnn_r50/latest.pth /
--format-only /
--options "jsonfile_prefix=./results"
```

### 학습 로그 결과 플롯
실제로 학습을 하면 여러 번 실행하고, 종료하고를 반복하기 때문에 많은 로그파일이 생긴다. 따라서 로그 파일을 열어 step별 생성되는 로그가 있는지를 확인해야한다.

<p align="center"><img src="https://user-images.githubusercontent.com/41941019/179204071-583d686c-75b9-439e-96bd-d40a8b1da86c.png"  width="100%"></p>

```
python mmdetection/tools/analysis_tools/analyze_logs.py /
plot_curve /
work_dirs/faster_rcnn_r50/log.json /
--keys bbox_mAP /
--legend bbox_mAP /
--out bbox_mAP.pdf
```


<br/>
## References
[1] Skywalk, 초보를 위한 정보이론 안내서 - KL divergence 쉽게 보기, https://hyunw.kim/blog/2017/10/27/KL_divergence.html      
[2] 하우론, [학부생의 딥러닝] GANs | WGAN, WGAN-GP : Wassestein GAN(Gradient Penalty), https://haawron.tistory.com/21?category=752293