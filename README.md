# Sign Board Text detection using Faster R-CNN
Generate synthetic dataset for object detection (Faster R-CNN) + evaluate on real data

### Methodology:
1. Create Synthetic Dataset comprising images and bounding box coordinates for Arabic and English texts.
2. Train a Faster RCNN model on the synthetic dataset
3. Collect a few real life examples of English and Arabic sign boards and evaluate the model

#### Generate images + bounding box dataset
```generate_synthetic_dataset.py```

#### Train on fake dataset
```train_fasterrcnn.py```

#### Evaluate on real dataset
```eval_fasterrcnn.py```

### Synthetic Dataset Generation: Steps
![Image of Steps](https://github.com/kvsnoufal/signboard_text_detection/tree/master/readme_images/steps.png)

![Image of Fake Data 1](https://github.com/kvsnoufal/signboard_text_detection/tree/master/readme_images/fakeimages.png)

### Evaluation on Real Data
![Image of Eval 2](https://github.com/kvsnoufal/signboard_text_detection/tree/master/readme_images/eval1.png)

![Image of Eval 1](https://github.com/kvsnoufal/signboard_text_detection/tree/master/readme_images/eval2.png)
