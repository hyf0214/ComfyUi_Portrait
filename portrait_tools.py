import numpy as np
import logging
from PIL import Image
import os
import base64
import requests
import torch
from torchvision import transforms
import torch
from typing import List
import oss2

from alibabacloud_facebody20191230.client import Client as facebody20191230Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_facebody20191230 import models as facebody_20191230_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
from viapi.fileutils import FileUtils


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> facebody20191230Client:
        """
        使用AK&SK初始化账号Client
        @return: Client
        @throws Exception
        """
        # 工程代码泄露可能会导致 AccessKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考。
        # 建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html。
        config = open_api_models.Config(
            # 必填，请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID。,
            # access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
            access_key_id="LTAI5tRZVTAGMPC2upJFs6Qe",
            # 必填，请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_SECRET。,
            # access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"]
            access_key_secret="8FoKcCQ2Jxbzc9INcQvOPe3EpTeT2o"
        )
        # Endpoint 请参考 https://api.aliyun.com/product/facebody
        config.endpoint = f"facebody.cn-shanghai.aliyuncs.com"
        return facebody20191230Client(config)

class PortraitTools(object):
    def __init__(self):
        self.image_type = [".jpg", ".jpeg", ".png", ".JPG"]
        self.max_size = (2000, 2000)
        self.logger = self.init_logger()
        self.oly_w = None
        self.oly_h = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                 "float_field": ("FLOAT", {
                    "default": 1,
                    "min": 0.1,
                    "max": 1,
                    "step": 0.1, #The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"

    OUTPUT_NODE = True

    CATEGORY = "PortraitTools"

    @staticmethod
    def get_log_level_from_env(env_key: str):
        log_level = os.environ.get(env_key, "DEBUG")
        log_level = log_level.upper()
        if log_level == "DEBUG":
            log_level = logging.DEBUG
        elif log_level == "WARNING":
            log_level = logging.WARNING
        elif log_level == "ERROR":
            log_level = logging.ERROR
        else:  # fallback to Info level
            log_level = logging.INFO
        return log_level

    def init_logger(self):
        logger = logging.getLogger("uvicorn")
        logger.handlers.clear()
        log_level = self.get_log_level_from_env("LOGGER_LOG_LEVEL")
        logger.setLevel(log_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(filename)s#%(funcName)s:%(lineno)d] %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def compress_image(self, input_image_path, output_image_path, quality=85):
        """
        压缩图片
        :param input_image_path: 输入图片路径
        :param output_image_path: 输出图片路径
        :param quality: 压缩质量（1-95）
        """
        img = Image.open(input_image_path)
        img.save(output_image_path, "JPEG", optimize=True, quality=quality)

    def check_and_compress_image(self, image_path, max_size_mb=3):
        """
        检查图片大小并在超过指定大小时压缩图片
        :param image_path: 图片路径
        :param max_size_mb: 最大允许的图片大小（MB）
        """
        max_size_bytes = 3 * 1024 * 1024
        if os.path.getsize(image_path) > int(max_size_bytes):
            self.logger.info(f"图片 {image_path} 大小超过 {max_size_mb}MB，正在压缩...")
            compressed_path = image_path.replace(".", "_compressed.")
            self.compress_image(image_path, compressed_path)
            self.logger.info(f"已压缩图片保存到 {compressed_path}")
        else:
            compressed_path = image_path
            self.logger.info(f"图片 {image_path} 大小在允许范围内，无需压缩。")
        return compressed_path

    def image2base64(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            # 根据图片类型添加适当的前缀，这里以JPEG为例
            image_type = image_path.split(".")[-1]
            return f"data:image/{image_type};base64,{encoded_string}"

    def smaller_resize_image(self, image):
        # 打开图像
        # 获取原始尺寸
        # 转换为 PIL.Image 对象
        original_size = image.size
        self.logger.info(f"Original size: {original_size}")

        # # 检查是否需要缩小
        # if original_size[0] > self.max_size[0] or original_size[1] > self.max_size[1]:
        #     # 按比例缩小图像
        #     img.thumbnail(self.max_size, Image.ANTIALIAS)
        # 计算新的尺寸，保持图像的宽高比
        new_path = "image_new_px.png"
        if image.size[0] > image.size[1]:
            ratio = self.max_size[0] / float(image.size[0])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))

            img_resized = image.resize(new_size, Image.LANCZOS)
            self.oly_w = new_size[0]
            self.oly_h = new_size[1]
        elif image.size[1] > image.size[0]:
            ratio = self.max_size[1] / float(image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            # 使用抗锯齿过滤器调整图像大小
            img_resized = image.resize(new_size, Image.LANCZOS)
            self.logger.info(f"Resized size: {original_size}")
            # 保存缩小后的图像
            self.oly_w = new_size[0]
            self.oly_h = new_size[1]
        else:
            img_resized = image
            self.oly_w = image.size[0]
            self.oly_h = image.size[1]
        img_resized.save(new_path)  # 保持原尺寸保存图像
        return new_path

    def image_oss_url(self, image_path):
        file_utils = FileUtils("LTAI5tRZVTAGMPC2upJFs6Qe", "8FoKcCQ2Jxbzc9INcQvOPe3EpTeT2o")
        oss_url = file_utils.get_oss_url(image_path, "jpg", True)
        return oss_url

    def resize_image(self, input_image_path):
        """
        将图片调整到指定大小

        :param input_image_path: 输入图片的路径
        :param output_image_path: 输出图片的路径
        :param size: 指定的新大小，格式为(宽度, 高度)
        """
        # 打开图片
        img = Image.open(input_image_path)
        size = (self.oly_w, self.oly_h)
        # 缩放图片
        resized_img = img.resize(size, Image.LANCZOS)  # 使用ANTIALIAS过滤器来平滑图片
        output_image_path = "image_restore.png"
        # 保存图片
        resized_img.save(output_image_path)
        os.remove(input_image_path)


        return output_image_path

    def facetidy_up(self, img_oss_url, contrast):
        client = Sample.create_client()
        face_tidyup_request = facebody_20191230_models.FaceTidyupRequest(
            shape_type=2,
            image_url=img_oss_url,
            strength=contrast,
        )
        runtime = util_models.RuntimeOptions()
        data = client.face_tidyup_with_options(face_tidyup_request, runtime)
        image_url = data.body.data.image_url
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        save_path = "./image_dispose.png"
        with open(save_path, "ab") as f:  # Use "ab" to append in binary mode
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path

    def process_image(self, image, float_field):
        # 选择第一个图像（如果批次大小 N > 1）
        image_tensor = image[0]
        img_np = (image_tensor * 255).round().numpy().astype(np.uint8)
        image = Image.fromarray(img_np, 'RGB')
        # image.show()

        new_path = self.smaller_resize_image(image)

        # 判断图片大小不超3MB
        compressed_path = self.check_and_compress_image(new_path)
        img_oss_url = self.image_oss_url(compressed_path)

        save_path = self.facetidy_up(img_oss_url, float_field)
        os.remove(compressed_path)

        # os.remove(save_path)
        # img = Image.open(save_path).convert('RGB')
        # transform = transforms.ToTensor()
        # img_tensor = transform(img)
        #
        # # 添加batch维度（如果需要）
        # img_tensor = img_tensor.unsqueeze(1)
        image = Image.open(save_path).convert('RGB')

        image_tensor = pil2tensor(image)

        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "PortraitTools": PortraitTools,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PortraitTools": "人像瘦脸",
}