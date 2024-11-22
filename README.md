## 整个项目的目录结构
``` bash
├── aaa_me
    - 存放一些自己写的备忘录，无意义
├── ab_get
    - 消融实验的代码
├── al_method
    - 为主动学习写的一些方法和工具
├── dataset_use
    - 这一文件夹存放一些lidc、exact09数据集裁剪、处理的代码
├── func
    - 分割网络的代码
├── test
└── train
```

## al_method文件夹的目录结构
``` bash
ALinAirway/al_method
├── active_learning_utils.py
    - 现已废弃,使用active_utils文件夹中的函数
├── active_utils
│   ├── cluster_tools.py
│   ├── dataset_process_tools.py
│   ├── embedding_tools.py
│   ├── file_tools.py
│   ├── random_crop.py
│   ├── select_tools.py
│   └── visualize_tools.py
├── get_emb.py
├── get_similarity.py
├── get_training_info.ipynb
└── select_pro.ipynb
```

## 使用方法
1. 初次使用时，通过get_emb.py获取数据集的embedding，供后续步骤使用(也可以实时进行，但是很慢)
2. 获取到embedding后，使用get_similarity.py获取embedding的相似度矩阵、uncertainy等
3. get_training_info.ipynb基于uncertainy生成训练集info，后续使用train_green.py进行训练
