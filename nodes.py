import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
import sys
current_node_dir = os.path.dirname(os.path.abspath(__file__)) 
custom_nodes_dir = os.path.dirname(current_node_dir)           
impact_pack_path = os.path.join(custom_nodes_dir, "ComfyUI-Impact-Pack")

if impact_pack_path not in sys.path:
    sys.path.insert(0, impact_pack_path)
impact_modules_path = os.path.join(impact_pack_path, "modules")
if impact_modules_path not in sys.path:
    sys.path.insert(0, impact_modules_path)

try:

    from modules.impact import core
    print("Fast-Mosaic: 成功桥接 Impact-Pack 核心库")
except ImportError:
    try:
        from impact.utils import core
        print("Fast-Mosaic: 通过备用路径成功桥接 Impact-Pack")
    except ImportError as e:
        print(f"Fast-Mosaic: 导入失败！错误信息: {e}")
import comfy.model_management
import torch.nn.functional as F


if not os.path.exists(impact_pack_path):
    print(f"!!! [Fast-Mosaic] 找不到 Impact-Pack 路径: {impact_pack_path}，请确保已安装该插件。")
import time
import hashlib

_FAST_MOSAIC_GLOBAL_STATE = {
    "last_hash": None
}

class BatchDirectoryLoader:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING" )
    RETURN_NAMES = ("image", "mask", "filename" )
    FUNCTION = "load_image"
    CATEGORY = "Fast-Mosaic"

    def load_image(self, folder_path, index):

        if not os.path.exists(folder_path):
            return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)), "no_folder", True)

        exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]
        files.sort()        
        
        total_files = len(files)
        target_file = files[index]
        full_path = os.path.join(folder_path, target_file)
        file_stat = os.stat(full_path)
        current_hash = hashlib.md5(f"{target_file}_{file_stat.st_size}".encode()).hexdigest()

        if _FAST_MOSAIC_GLOBAL_STATE["last_hash"] == current_hash:
            _FAST_MOSAIC_GLOBAL_STATE["last_hash"] = None 
            raise StopIteration("All images processed.")

        _FAST_MOSAIC_GLOBAL_STATE["last_hash"] = current_hash
        
        if total_files == 0:
            raise FileNotFoundError(f"所选路径下没有图片: {folder_path}")

        img = Image.open(full_path).convert("RGB")
        img = ImageOps.exif_transpose(img) 
        
        image_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None, ...] 

        mask = torch.zeros((image_np.shape[0], image_np.shape[1]), dtype=torch.float32)

        return (image_tensor, mask, target_file )




class MosaicBlurSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["Mosaic", "Gaussian Blur"], {"default": "Mosaic"}),

                "bbox_detector": ("BBOX_DETECTOR", ),
                "bbox_threshold":("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 10, "min": -255, "max": 255, "step": 1}),
                "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": 2048, "step": 1, "default": 10}),

                "intensity": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}), 
                "feather_amount": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "output_directory": ("STRING", {"default": "output/processed"}),
                "filename_prefix": ("STRING", {"default": "result"}),
                "save_format": (["png", "jpg"], {"default": "png"}),
                "preview_interval": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}), 
                "index": ("INT", {"default": 0}), 
                "original_file_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "sam_model_opt": ("SAM_MODEL", ),
                "segm_detector_opt": ("SEGM_DETECTOR", ),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_and_save"
    CATEGORY = "Fast-Mosaic"
    OUTPUT_NODE = True

    def process_and_save(self, image,mode,bbox_detector,bbox_threshold,
                         bbox_dilation,bbox_crop_factor,drop_size,
                         intensity, feather_amount, 
                         output_directory, filename_prefix, save_format, 
                         preview_interval, index, original_file_path,
                         sam_model_opt=None,segm_detector_opt=None):
        
        mask_list = []
        
        # 遍历批次中的每一张图Iterate through each image in the batch.
        for i in range(len(image)):
            curr_img = image[i:i+1]

            segs = bbox_detector.detect(curr_img, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
            
            if segm_detector_opt is not None:
                segs = segm_detector_opt.detect(curr_img, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)


            if sam_model_opt is not None:
                
                segs = core.enhance_mask_with_sam(curr_img, segs, sam_model_opt, 
                                                 "center-1", 0, 0.20, 0, 0.7, True)

            if len(segs[1]) > 0:

                combined_mask = core.segs_to_combined_mask(segs)
            else:
                
                combined_mask = torch.zeros((curr_img.shape[1], curr_img.shape[2]))

            mask_list.append(combined_mask)

        final_mask = torch.stack(mask_list, dim=0)

        batch_size = image.shape[0]
        processed_images = []
        ui_images = []
        
        base_output_dir = folder_paths.get_output_directory()
        safe_output_dir = os.path.basename(output_directory)
        final_path = os.path.join(base_output_dir, safe_output_dir)
        
        if not os.path.exists(final_path):
            os.makedirs(final_path)

        processed_images = []

        for i in range(batch_size):
            # --- 图像准备 Image Preparation---
            curr_img = image[i]

            img_shape = curr_img.shape
            h, w = img_shape[0], img_shape[1]
            
            
            if curr_img.dim() == 3:
                
                curr_img = curr_img.unsqueeze(0).permute(0, 3, 1, 2)
            elif curr_img.dim() == 2:
                
                curr_img = curr_img.unsqueeze(0).unsqueeze(0)

            curr_mask = final_mask[i]

            if curr_mask.dim() == 2:
                curr_mask = curr_mask.unsqueeze(0).unsqueeze(0)
            elif curr_mask.dim() == 3:
                curr_mask = curr_mask.unsqueeze(0)
            elif curr_mask.dim() == 4 and curr_mask.shape[1] > 1:
                curr_mask = curr_mask.permute(0, 3, 1, 2)

            
            if feather_amount > 0:
                kernel_size = feather_amount * 2 + 1
                
                curr_mask = F.avg_pool2d(curr_mask, kernel_size, stride=1, padding=feather_amount)
            if feather_amount > 0:
                kernel_size = feather_amount * 2 + 1
                
                curr_mask = F.avg_pool2d(curr_mask, kernel_size, stride=1, padding=feather_amount)


            if mode == "Mosaic":
                sw, sh = max(1, w // intensity), max(1, h // intensity)
                low_res = F.interpolate(curr_img, size=(sh, sw), mode='bilinear')
                processed_layer = F.interpolate(low_res, size=(h, w), mode='nearest')
            else:

                kernel_size = intensity * 2 + 1
                processed_layer = F.avg_pool2d(curr_img, kernel_size, stride=1, padding=intensity)

            final_img = curr_img * (1.0 - curr_mask) + processed_layer * curr_mask
            
            final_img_out = final_img.permute(0, 2, 3, 1).squeeze(0)

            if original_file_path:
                orig_name = os.path.splitext(os.path.basename(original_file_path))[0]
                save_name = f"{filename_prefix}_{orig_name}.{save_format}"
            else:
                save_name = f"{filename_prefix}_{index + i}.{save_format}"
            
            save_path = os.path.join(output_directory, save_name)

            img_np = (final_img_out.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img.save(save_path, quality=95 if save_format == "jpg" else None)
            
            if preview_interval > 0 and i % preview_interval == 0:

                preview_img = (final_img_out.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                preview_pil = Image.fromarray(preview_img)
                
                # 将图片存入 ComfyUI 的临时目录，供预览节点读取Save the image to ComfyUI's temporary directory for the preview node to read.
                preview_filename = f"preview_{i}.png"
                preview_path = os.path.join(folder_paths.get_temp_directory(), preview_filename)
                preview_pil.save(preview_path)
                
                ui_images.append({
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp"
                })   
           
            processed_images.append(final_img_out)

        output_stack = torch.stack(processed_images)

        return {
            "ui": {"images": ui_images}, 
            "result": (output_stack,)
            }

   