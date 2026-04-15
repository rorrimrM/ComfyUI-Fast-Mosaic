* 支持对图片进行马赛克处理或高斯模糊，可以调节马赛克大小(intensity)和羽化大小(feather_amount)
批处理请开启Auto Queue或秋叶版的运行(实时)，每次记得重置primitive节点的整数，当出现报错弹窗代表处理完成，结果在input中

* 使用前请务必确保comfyui的custom_nodes文件下存在[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)插件(确保不修改插件名称)

* 下载`requirements.txt`的python库

* 常用的yolo模型以及segm模型：
https://huggingface.co/24xx/segm/tree/main
https://huggingface.co/Bingsu/adetailer/tree/main
https://huggingface.co/ashllay/YOLO_Models/tree/main/bbox
https://huggingface.co/ashllay/YOLO_Models/tree/main/segm

* 将yolo模型放在`comfyui/models/ultralytics/bbox`下
将segm模型放在`comfyui/models/ultralytics/segm`下


* Supports applying mosaic effects or Gaussian blur to images; you can adjust the mosaic intensity and feather amount.
For batch processing, please enable "Auto Queue" or the "Run (Real-time)" option (in the QiuYe edition).Remember to reset the integer value of the  
 primitive node every time. When an error pop-up window appears, it indicates that the processing is complete; the results can be found in the "input" folder.

* Before use, please ensure that the [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) plugin is present within the  
`custom_nodes` directory of your ComfyUI installation (ensure that the plugin's folder name remains unchanged).

* Download the Python libraries listed in the `requirements.txt` file.

* Commonly used YOLO and segmentation models:
https://huggingface.co/24xx/segm/tree/main
https://huggingface.co/Bingsu/adetailer/tree/main
https://huggingface.co/ashllay/YOLO_Models/tree/main/bbox
https://huggingface.co/ashllay/YOLO_Models/tree/main/segm

* Place the YOLO models in the `comfyui/models/ultralytics/bbox` directory.
Place the segmentation models in the `comfyui/models/ultralytics/segm` directory.