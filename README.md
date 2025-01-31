
# Blackspot-in-Floor Detection (Mask R-CNN)

This repository contains a Detectron2-based pipeline for **segmenting floors** in an image and subsequently detecting **“blackspots”** (dark rug, mat, or spill) **only** if they lie on top of the floor area. This can be especially useful for:

- **Hazard detection**: identifying wet or slippery areas on floors.  
- **Robotics**: detecting irregular floor surfaces or potential obstacles.  
- **Assisted living**: noticing spills or trip hazards in real time.

  
## Key Features

1. **Two-Category Segmentation**  
   - **Floors** (ID=0)
   - **blackspot** (ID=1)

2. **Unified Floor Category**  
   If your dataset had multiple “floor” categories, we unify them into a single one. This ensures a single mask for all floor instances.

3. **Blackspot-on-Floor Overlap**  
   After detecting floors and blackspots, the code checks for overlaps. Only blackspots overlapping with the floor region get flagged visually (drawn with bounding boxes).

4. **Mask R-CNN**  
   - A powerful, state-of-the-art instance segmentation model (Detectron2).
   - Outputs bounding boxes **and** segmentation masks.

5. **COCO-Style Evaluation**  
   - Uses **mAP**, **AR**, and other standard metrics via COCOEvaluator.  
   - Provides both **bounding box** and **mask** (segmentation) evaluation results.


---

## Example Results

Below is an example detection result.  
A **red bounding box** highlights a “blackspot” region that overlaps with the **floor**:

![Blackspot Over Floor Detection - Demo](docs/blackspot_demo.png)  
*(In this image, the black rug on the floor is automatically detected, and the bounding box is drawn only if the blackspot mask overlaps with the floor mask.)*

---

## Repository Contents

- **`Example.py`** (or `Untitled.ipynb`): Contains the entire pipeline to:
  1. Load and unify the dataset (merging duplicate floor categories).
  2. Register the dataset in Detectron2.
  3. Configure and train Mask R-CNN on the new categories.
  4. Evaluate the model with COCO metrics.
  5. Run inference on a sample test image.

- **`train/`, `valid/`, `test/`**: Example dataset folders containing images and COCO-format JSON annotations (`_annotations.coco.json`).

- **`output_floor_blackspot/`**: The default output directory where training artifacts (model checkpoints, logs, etc.) are saved.

- **`utils/`**: Utility scripts for data loading, annotation unification, etc.

- **`Floor Blackspot Detect.v3i.coco-segmentation.zip`**: The dataset (COCO segmentation style) if included.

---

## Why This Model Is Effective

1. **Focus on Overlap**  
   Many segmentation models can find floors and blackspots separately. Here, we specifically highlight blackspots **only** if they lie on a **floor**. This post-processing step (merging masks) reduces false positives (black areas not on the floor).

2. **Mask R-CNN’s Versatility**  
   - It handles complex shapes (like irregular floor boundaries).
   - The separate mask head provides higher accuracy for segmenting distinct objects (e.g., black rugs, spills).

3. **Unified Floor Category**  
   - Some datasets label floor categories inconsistently (e.g., “Wood Floor” = 0, “Tiled Floor” = 1).  
   - We merge them into one “Floors” category, simplifying training and inference.

4. **COCO Metrics**  
   - Standard, well-known metrics for measuring bounding box **and** mask AP across different IoU thresholds.  
   - Quick insight into how well the model generalizes to new floor types and blackspots.


---

## Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/BlackSpot_FloorDetection.git
   cd BlackSpot_FloorDetection
   ```

2. **Create (and activate) a Conda or Python Virtual Environment**  
   ```bash
   conda create -n floor_blackspot python=3.10 -y
   conda activate floor_blackspot
   # or python -m venv venv and source venv/bin/activate
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   Ensure that Detectron2 is installed. If needed:
   ```bash
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

4. **Check GPU Access (optional)**  
   If you have a CUDA-enabled device, verify with:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   If `True`, the code will run with `cfg.MODEL.DEVICE = "cuda:0"`.

---

## Data Preparation

1. **COCO-Style Dataset**  
   - Folders named `train/`, `valid/`, `test/` with images inside.
   - Corresponding `_annotations.coco.json` in each folder describing polygons (segmentation) for floors and blackspots.

2. **Merging Floor Categories**  
   - If your dataset has multiple floor IDs (e.g., 0 and 1 both labeled “Floors”), the scripts unify them into a single ID=0.  
   - The “blackspot” category gets ID=1.

3. **Registering**  
   - `register_unified_floors_dataset(name, json_file, image_root)` calls an internal function that:
     1. **Loads** your original JSON.
     2. **Re-maps** floor categories to ID=0 and blackspot to ID=1.
     3. **Registers** it with Detectron2’s `DatasetCatalog`.

*(All of this logic is in the example script.)*

---

## Training

1. **Set the Model Config**  
   In `Example.py` (or `Untitled.ipynb`), we do:
   ```python
   cfg = get_cfg()
   cfg.merge_from_file(
       model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
   )
   cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # [Floors, blackspot]
   cfg.DATASETS.TRAIN = ("floors_train",)
   cfg.DATASETS.TEST = ("floors_val",)
   cfg.MODEL.DEVICE = "cuda:0"         # or "cpu"
   ...
   trainer = DefaultTrainer(cfg)
   trainer.resume_or_load(resume=False)
   trainer.train()
   ```

2. **Monitor Training**  
   - Output logs appear in the console.
   - By default, final weights get saved in `./output_floor_blackspot/`.

---

## Evaluation

1. **COCOEvaluator**  
   ```python
   evaluator = COCOEvaluator("floors_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
   test_loader = build_detection_test_loader(cfg, "floors_test")
   test_results = inference_on_dataset(predictor.model, test_loader, evaluator)
   print("Test metrics:", test_results)
   ```
   - Gives bounding box AP, mask AP, and per-category results (Floors vs. blackspot).

2. **Sample Output** (truncated):  
   ```
   Average Precision (AP) [bbox] = 0.354
   Average Precision (AP) [segm] = 0.353
   ...
   Per-category AP for [bbox]:
      Floors:    55.50
      blackspot: 15.39
   ...
   ```
   This indicates floors are detected quite reliably, while blackspot detection AP is around 15–20%. Further data or training can improve blackspot performance.

---

## Inference / Demo

Use the final trained model to **detect blackspots over floors** in any new image:

```python
# Once training is done, load the final weights:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
predictor = DefaultPredictor(cfg)

# Inference on a test image
im = cv2.imread("path/to/your_image.jpg")
outputs = predictor(im)
instances = outputs["instances"]

# Separate floor masks vs. blackspot masks
floor_masks      = []
blackspot_masks  = []
for cls_id, m in zip(instances.pred_classes.cpu().numpy(),
                     instances.pred_masks.cpu().numpy()):
    if cls_id == 0:  # Floors
        floor_masks.append(m)
    else:            # blackspot
        blackspot_masks.append(m)

# Combine floor masks into one big floor region
combined_floor_mask = np.any(floor_masks, axis=0) if floor_masks else None

# Identify blackspot bounding boxes only where overlap with floor occurs
# (Implementation in the example script)
...
```

When you display the result (using OpenCV or matplotlib), you’ll see bounding boxes only on blackspot regions that lie on top of the detected floor.

---

## Why This Matters

- **Safety**: Promptly detect unusual floor conditions in real-world environments.  
- **Accuracy**: Mask R-CNN’s segmentation approach outperforms simple bounding-box detectors for tasks like mat/spill detection.  
- **Scalability**: You can add more categories (like “objects on the floor”) if needed, but the pipeline stays the same.  

---

## Troubleshooting & Tips

1. **Check Image Paths**  
   - Make sure your `train/`, `valid/`, `test/` folder structure and `*_annotations.coco.json` references the correct image filenames.

2. **Multiple Floor Categories**  
   - Confirm that your original annotation IDs for “Floors” unify into ID=0 (the script does this automatically).

3. **Low Blackspot AP?**  
   - You may need more training images with varied blackspots (lighting conditions, shapes, colors).  
   - Adjust `cfg.SOLVER.MAX_ITER`, or tweak the learning rate and data augmentation.

4. **GPU vs. CPU**  
   - GPU is recommended (NVIDIA CUDA).  
   - If you only have CPU, training will be slower—make sure you change `cfg.MODEL.DEVICE = "cpu"`.

---

## Conclusion

With this repository, you can:
- **Train a Mask R-CNN** model to identify **floors** vs. **blackspots** in a single pass.  
- **Highlight** blackspots **only** if they lie on top of floors.  
- Use standard **COCO metrics** to measure performance.  

