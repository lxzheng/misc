# TensorFlow Object Detection API configure file

## centernet\_mobilenetv2\_fpn\_kpts配置文件分析

```
model {
  center_net {
    num_classes: 1
    feature_extractor {
      type: "mobilenet_v2_fpn_sep_conv"
    }
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 512
        max_dimension: 512
        pad_to_max_dimension: true
      }
    }
    use_depthwise: true
    object_detection_task {
      task_loss_weight: 1.0
      offset_loss_weight: 1.0
      scale_loss_weight: 0.1
      localization_loss {
        l1_localization_loss {
        }
      }
    }
    object_center_params {
      object_center_loss_weight: 1.0
      classification_loss {
        penalty_reduced_logistic_focal_loss {
          alpha: 2.0
          beta: 4.0
        }
      }
      min_box_overlap_iou: 0.7
      max_box_predictions: 20
    }
```

模型配置如下：

1.  使用了CenterNet模型，类别数为1。
2.  使用Mobilenet V2 FPN SEP Conv作为特征提取器。
3.  图像重新调整的尺寸为512×512，且保持比例。
4.  使用深度卷积（depthwise）。
5.  设置了目标检测任务，其中定位损失使用L1定位损失。
6.  设置了目标中心参数，其中分类损失使用Penalty Reduced Logistic Focal Loss，IoU阈值为0.7，最多预测20个盒子。
7.  指定了关键点标签映射路径。
8.  设置了关键点估计任务，任务名为“human\_pose”，其中定位损失使用L1定位损失，分类损失使用Penalty Reduced Logistic Focal Loss，指定了特定的关键点类名，并为每个关键点设置了标准偏差值。

*   `model {...}`：定义模型的配置。
*   `center_net {...}`：定义使用的网络架构（CenterNet）的配置。
    *   `num_classes: 1`：定义类别数量为1。
*   `feature_extractor {...}`：定义使用的特征提取器类型（mobilenet\_v2\_fpn\_sep\_conv）。
*   `image_resizer {...}`：定义图像的调整方式。
*   `keep_aspect_ratio_resizer {...}`：定义使用保持宽高比的方式进行调整。
    *   `min_dimension: 512`：定义最小尺寸为512。
    *   `max_dimension: 512`：定义最大尺寸为512。
    *   `pad_to_max_dimension: true`：定义图像是否要填充到最大尺寸。
*   `use_depthwise: true`：定义是否使用深度方向卷积。
*   `object_detection_task {...}`：定义目标检测任务的配置。
*   `task_loss_weight: 1.0`：定义目标检测任务的权重。
*   `offset_loss_weight: 1.0`：定义偏移量损失函数的权重。
*   `scale_loss_weight: 0.1`：定义比例损失函数的权重。
*   `localization_loss {...}`：定义定位损失函数。
    *   `l1_localization_loss {...}`：定义使用L1损失函数。
*   `object_center_params {...}`：定义对象中心参数的配置。
    *   `object_center_loss_weight: 1.0`：定义对象中心损失函数的权重。这是一个权重参数，用于控制对物体中心的损失的影响。
    *   `classification_loss`: 这是分类损失的配置，包括使用的损失函数以及其相关参数。
        *   `min_box_overlap_iou`: 这是一个阈值，用于控制最小的预测框重叠 IOU。
        *   `max_box_predictions`: 这是一个最大预测框数量的限制。

```
keypoint_label_map_path: "PATH_TO_BE_CONFIGURED/label_map.txt"
    keypoint_estimation_task {
      task_name: "human_pose"
      task_loss_weight: 1.0
      loss {
        localization_loss {
          l1_localization_loss {
          }
        }
        classification_loss {
          penalty_reduced_logistic_focal_loss {
            alpha: 2.0
            beta: 4.0
          }
        }
      }
      keypoint_class_name: "/m/01g317"
      keypoint_label_to_std {
        key: "left_ankle"
        value: 0.89
      }
```

*   `keypoint_label_map_path`：是文件的路径，指向某个 label\_map.txt，应该在配置这个路径之前将它修改为合适的值。 label\_map.txt包含了标签的映射关系，格式为 "标签名称: 标签编号"
*   `keypoint_estimation_task`：描述了一个任务，即骨架关键点的估计。这个部分定义了键点估计的任务，它包括任务的名称，任务的损失权重，损失函数的定义，键点的分类名称，键点标签到标准的映射，键点回归损失的权重，键点热图损失的权重，键点偏移损失的权重，偏移的峰值半径，是否为每个键点分别计算偏移等。
    *   `task_name`：任务的名称，为 "human\_pose"
    *   `task_loss_weight`：任务的损失权重，设置为 1.0
    *   `loss`：描述了任务的损失函数
        *   `localization_loss`：位置损失，使用 L1 本地化损失函数
        *   `classification_loss`：分类损失，使用 penalty\_reduced\_logistic\_focal\_loss 损失函数。
            *   `alpha`：设置为 2.0
            *   `beta`：设置为 4.0
    *   `keypoint_class_name`：关键点类别的名称，为 "/m/01g317"
    *   `keypoint_label_to_std`：关键点标签和标准差的映射，如 "left\_ankle" 映射到 0.89

```
      keypoint_regression_loss_weight: 0.1
      keypoint_heatmap_loss_weight: 1.0
      keypoint_offset_loss_weight: 1.0
      offset_peak_radius: 3
      per_keypoint_offset: true
```

这些参数用于配置关键点回归任务。

*   keypoint\_regression\_loss\_weight：关键点回归损失的权重，用于调整损失函数中关键点回归损失的影响。
*   keypoint\_heatmap\_loss\_weight：关键点热图损失的权重，用于调整损失函数中关键点热图损失的影响。
*   keypoint\_offset\_loss\_weight：关键点偏移损失的权重，用于调整损失函数中关键点偏移损失的影响。
*   offset\_peak\_radius：关键点偏移损失计算时的邻域半径。
*   per\_keypoint\_offset：是否对每个关键点单独计算偏移损失。

```
train_config {
  batch_size: 512
  data_augmentation_options {
    random_horizontal_flip {
      keypoint_flip_permutation: 0
      keypoint_flip_permutation: 2
      keypoint_flip_permutation: 1
      keypoint_flip_permutation: 4
      keypoint_flip_permutation: 3
      keypoint_flip_permutation: 6
      keypoint_flip_permutation: 5
      keypoint_flip_permutation: 8
      keypoint_flip_permutation: 7
      keypoint_flip_permutation: 10
      keypoint_flip_permutation: 9
      keypoint_flip_permutation: 12
      keypoint_flip_permutation: 11
      keypoint_flip_permutation: 14
      keypoint_flip_permutation: 13
      keypoint_flip_permutation: 16
      keypoint_flip_permutation: 15
    }
  }
```

这是关于训练的配置参数，这个部分定义了训练配置，包括批次大小，数据增强选项（随机水平翻转，随机裁剪，随机放缩，随机亮度，随机对比度），总训练步数，学习率调度策略等。其中：

*   `batch_size` 表示每一批的样本数量
*   `data_augmentation_options` 表示数据增强的选项，在这个例子中是随机水平翻转，并且列出了对关键点的顺序翻转。
    *   `keypoint_flip_permutation` 列出了在翻转图像时要进行交换的关键点的位置，以使得在翻转后关键点位置不变。在图像翻转时，根据配置中的 "keypoint\_flip\_permutation"，关键点位置是按照如下顺序交换的：0 和 2、1 和 4、3 和 6、5 和 8、7 和 10、9 和 12、11 和 14、13 和 16、15。

```
  data_augmentation_options {
    random_patch_gaussian {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_aspect_ratio: 0.5
      max_aspect_ratio: 1.7
      random_coef: 0.25
    }
  }
  data_augmentation_options {
    random_adjust_hue {
    }
  }
  data_augmentation_options {
    random_adjust_contrast {
    }
  }
  data_augmentation_options {
    random_adjust_saturation {
    }
  }
  data_augmentation_options {
    random_adjust_brightness {
    }
  }
  data_augmentation_options {
    random_absolute_pad_image {
      max_height_padding: 200
      max_width_padding: 200
      pad_color: 0.0
      pad_color: 0.0
      pad_color: 0.0
    }
  }
```

这些是模型训练的数据增强选项。

*   `random_patch_gaussian`：以一定概率在图像中随机添加高斯噪声。
*   `random_crop_image`：以一定概率对图像进行随机裁剪，保证裁剪后的图像的宽高比在 \[0.5, 1.7\] 之间，并且随机裁剪因子为 0.25。
*   `random_adjust_hue`：以一定概率随机调整图像的色调。
*   `random_absolute_pad_image`：以一定概率对图像进行随机填充，使得填充后的图像的最大高度为 200，最大宽度为 200，填充的颜色为 (0, 0, 0)。

![](https://chat.openai.com/_next/image?url=https%3A%2F%2Flh3.googleusercontent.com%2Fa%2FAEdFTp6u_d0gLMsFG3WR0cF2wFcJPj3iNuI_ikzsvjEn%3Ds96-c&w=64&q=75)

```
  optimizer {
    adam_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 5e-3
          total_steps: 300000
          warmup_learning_rate: 1e-4
          warmup_steps: 5000
        }
      }
    }
    use_moving_average: false
  }
  num_steps: 300000
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: ""
}
train_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/label_map.txt"
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/train2017-?????-of-00256.tfrecord"
  }
  filenames_shuffle_buffer_size: 256
  num_keypoints: 17
}
```

*   `optimizer` 配置了模型的优化算法。这里使用的是Adam算法，学习率是以`cosine_decay_learning_rate`的形式变化的，初始学习率为5e-3，总步数为300000步，预热步数为5000步，预热学习率为1e-4。
*   `num_steps` 设置了模型训练的步数。
*   `max_number_of_boxes` 设置了每张图像中可以识别的最大目标数量。
*   `unpad_groundtruth_tensors` 用于控制在图像比例缩放后，是否需要重新对齐边界框，默认为False。
*   `fine_tune_checkpoint_type` 决定了是否使用预训练的检查点来微调模型，这里为空字符串，不使用预训练的检查点。
*   `train_input_reader` 中的配置指定了数据读取的方式。这里，图像的标签信息在`label_map_path`指定的文件中，输入图像数据使用了TFRecord格式，输入数据在多个tfrecord文件中，这些文件的路径需要在输入读取器配置中指定。读取文件时打乱顺序的缓存大小是256，每张图像有17个关键点。

```
eval_config {
  num_visualizations: 10
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  min_score_threshold: 0.2
  max_num_boxes_to_visualize: 20
  batch_size: 1
  parameterized_metric {
    coco_keypoint_metrics {
      class_label: "person"
      keypoint_label_to_sigmas {
        key: "left_ankle"
        value: 0.089
      }
......
  keypoint_edge {
    start: 0
    end: 1
  }
  keypoint_edge {
    start: 0
    end: 2
  }
......
eval_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/label_map.txt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/val2017-?????-of-00032.tfrecord"
  }
  num_keypoints: 17
}
```

这些参数是TensorFlow Object Detection API模型的评估配置。

*   `eval_config`：描述了模型评估的配置，包括以下几点：
    *   `num_visualizations`：评估过程中可视化的图像数量。
    *   `metrics_set`：评估所使用的指标的集合。
    *   `use_moving_averages`：是否使用平均移动的模型权重来评估模型。
    *   `min_score_threshold`：进行评估时，过滤掉置信度较低的检测结果的阈值。
    *   `max_num_boxes_to_visualize`：评估过程中最多可视化的检测结果的数量。
    *   `batch_size`：每批评估的图像数量。
    *   `parameterized_metric`：评估需要使用的参数化指标。
        *   `coco_keypoint_metrics`: 指定了使用 COCO 格式的关键点评估。
        *   `class_label`: 指定了要评估的目标类别，在这种情况下为 "person"。
        *   `keypoint_label_to_sigmas` :一个字典，用于指定每个关键点的允许误差范围。在这里，对于 "left\_ankle" 关键点，误差范围为 0.089。
    *   `keypoint_edge`：评估关键点检测结果时需要使用的边界。
*   `eval_input_reader`：描述了用于评估的数据的读取配置，包括以下内容：
    *   `label_map_path`：指向标签映射文件的路径。
    *   `shuffle`：是否打乱评估数据的顺序。
    *   `num_epochs`：评估数据的迭代次数。
    *   `tf_record_input_reader`：描述读取评估数据的方式，使用TFRecord文件。
    *   `num_keypoints`：关键点数量。
