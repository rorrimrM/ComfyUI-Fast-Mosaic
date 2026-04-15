from .nodes import BatchDirectoryLoader,MosaicBlurSaver

NODE_CLASS_MAPPINGS = {
   "Batch Directory Loader": BatchDirectoryLoader,
   "MosaicBlurSaver": MosaicBlurSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Batch Directory Loader": "Batch Directory Loader",
   "MosaicBlurSaver": "MosaicBlurSaver"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']