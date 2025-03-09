import os
import cv2
import numpy as np
import tempfile
import gradio as gr
import torch
from PIL import Image

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

########################################
# 1. SETUP DETECTRON2 MASK R-CNN MODEL
########################################
def load_maskrcnn_model(weights_path, device="cuda", threshold=0.5):
    cfg = get_cfg()
    # Load the base config from the COCO instance segmentation model zoo.
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    # Set the number of classes: [Floors, blackspot]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # Set the detection threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    # Set the path to your trained weights
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device

    return DefaultPredictor(cfg)

########################################
# 2. POST-PROCESSING FUNCTIONS
########################################
def postprocess_masks(im, instances, show_floor=True, show_blackspot=True):
    """
    Extract floor and blackspot masks from the model's Instances,
    process them according to display preferences.
    """
    height, width = im.shape[:2]
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_masks = instances.pred_masks.cpu().numpy()
    
    # Create empty masks
    combined_floor_mask = np.zeros((height, width), dtype=bool)
    final_blackspot = np.zeros((height, width), dtype=bool)
    
    # Process masks based on class
    for cls_id, mask in zip(pred_classes, pred_masks):
        if cls_id == 0 and show_floor:  # Floor class
            combined_floor_mask |= mask
        elif cls_id == 1 and show_blackspot:  # Blackspot class
            final_blackspot |= mask
    
    # If we want to constrain blackspots to only appear on floors
    if show_blackspot and not show_floor:
        # We still need the floor mask for constraining blackspots
        floor_masks = []
        for cls_id, mask in zip(pred_classes, pred_masks):
            if cls_id == 0:
                floor_masks.append(mask)
                
        if floor_masks:
            all_floors = np.any(floor_masks, axis=0)
            # Only keep blackspots that overlap with floor
            final_blackspot = final_blackspot & all_floors

    return combined_floor_mask.astype(np.uint8), final_blackspot.astype(np.uint8)

def overlay_masks(im, floor_mask, blackspot_mask, show_floor=True, show_blackspot=True, 
                 floor_color=(0, 255, 0), blackspot_color=(0, 0, 255), alpha=0.5):
    """
    Overlay the floor and blackspot masks on the input image.
    """
    result = im.copy()
    
    # Create a blank overlay image
    overlay = np.zeros_like(im)
    
    # Apply floor mask if requested
    if show_floor:
        overlay[floor_mask > 0] = floor_color
    
    # Apply blackspot mask if requested (will override floor color)
    if show_blackspot:
        overlay[blackspot_mask > 0] = blackspot_color
    
    # Blend the overlay with the original image
    result = cv2.addWeighted(im, 1.0, overlay, alpha, 0)
    return result

########################################
# 3. INFERENCE FUNCTION
########################################
def inference_on_image(image_path, threshold, show_floor, show_blackspot):
    """
    Run inference with the specified prediction threshold and display options.
    """
    # Read the image using OpenCV
    im = cv2.imread(image_path)
    if im is None:
        return "Error: could not read image!", None
    
    # Create a predictor with the specified threshold
    threshold = float(threshold)
    weights_path = "./output_floor_blackspot/model_0004999.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model with the requested threshold
    predictor = load_maskrcnn_model(weights_path, device, threshold)
    
    # Run inference with Mask R-CNN
    outputs = predictor(im)
    instances = outputs["instances"]
    
    # Post-process to extract masks according to display preferences
    floor_mask, blackspot_mask = postprocess_masks(im, instances, show_floor, show_blackspot)
    
    # Create an overlay of the masks on the original image
    result_overlay = overlay_masks(im, floor_mask, blackspot_mask, 
                                  show_floor, show_blackspot)
    result_overlay_rgb = cv2.cvtColor(result_overlay, cv2.COLOR_BGR2RGB)
    
    # Calculate statistics
    blackspot_area = int(blackspot_mask.sum())
    floor_area = int(floor_mask.sum())
    
    # Create a summary report
    text_msg = [f"Prediction threshold: {threshold:.2f}"]
    
    if show_floor:
        text_msg.append(f"Floor area: {floor_area} pixels")
    
    if show_blackspot:
        text_msg.append(f"Blackspot area: {blackspot_area} pixels")
        if floor_area > 0 and show_floor:
            percentage = (blackspot_area / floor_area) * 100
            text_msg.append(f"Blackspot coverage: {percentage:.2f}% of floor area")
    
    # Count detections
    num_detections = len(instances)
    if num_detections > 0:
        floor_count = sum(1 for cls in instances.pred_classes.cpu().numpy() if cls == 0)
        blackspot_count = sum(1 for cls in instances.pred_classes.cpu().numpy() if cls == 1)
        text_msg.append(f"Detected {floor_count} floor regions and {blackspot_count} blackspot regions")
    
    return "\n".join(text_msg), Image.fromarray(result_overlay_rgb)

########################################
# 4. GRADIO APP
########################################
def create_demo_app():
    with gr.Blocks(title="Blackspot Detection Demo") as demo:
        gr.Markdown("## Mask R-CNN Blackspot & Floor Segmentation")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Input Image", type="filepath")
                
                with gr.Row():
                    threshold_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Prediction Threshold"
                    )
                
                with gr.Row():
                    show_floor_checkbox = gr.Checkbox(
                        value=True, label="Show Floor Detections"
                    )
                    show_blackspot_checkbox = gr.Checkbox(
                        value=True, label="Show Blackspot Detections"
                    )
                
                submit_button = gr.Button("Run Inference")
            
            with gr.Column():
                text_output = gr.Textbox(label="Analysis Results")
                overlay_output = gr.Image(label="Segmentation Visualization")
        
        # Connect components to the inference function
        submit_button.click(
            fn=inference_on_image,
            inputs=[image_input, threshold_slider, show_floor_checkbox, show_blackspot_checkbox],
            outputs=[text_output, overlay_output]
        )
    
    return demo

########################################
# 5. MAIN ENTRY POINT
########################################
if __name__ == "__main__":
    print("Initializing Blackspot Detection App...")
    weights_path = "./output_floor_blackspot/model_0004999.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model will load from: {weights_path}, device={device}")
    
    # Create and launch the Gradio interface
    demo = create_demo_app()
    # Launch the app with a sharable link
    demo.queue().launch(show_api=False, show_error=True, share=True)
