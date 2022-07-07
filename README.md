# military_event_detection
#### 高鲁棒性要求下的领域事件检测任务 [比赛官网](https://www.datafountain.cn/competitions/561/ranking?isRedance=0&sch=1917)

1. Version 1 (master):
- span-based seqence labeling + Dice Loss;
2. Version 2 (trick_v1):
- span-based seqence labeling + Weighted CE Loss → work;
3. Version 3 (RDropout):
- add: R-Dropout(para: $\alpha$) → loacl f1 up;
- del: Dice Loss;
- try: Focal Loss → not work;
