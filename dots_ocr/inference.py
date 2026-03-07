"""
仅能从PDF提取文字，完整提取(包含图片)请采用parser.py
"""

from openai import OpenAI
import requests
import os
from PIL import Image
from dots_ocr.utils.doc_utils import load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.image_utils import PILimage_to_base64
from argparse import ArgumentParser

import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))


def inference_with_vllm(
        image,
        prompt,
        protocol="http",
        ip="localhost",
        port=8000,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='rednote-hilab/dots.ocr',
        system_prompt=None,
):
    addr = f"{protocol}://{ip}:{port}/v1"
    client = OpenAI(api_key="{}".format(os.environ.get("API_KEY", "0")), base_url=addr)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}
                # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p)
        response = response.choices[0].message.content
        return response
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        return None


def convert_image_to_markdown(images, file_name_without_ext):
    if isinstance(images, list):
        for idx, img in enumerate(images):
            response = inference_with_vllm(
                img,
                prompt,
                ip="localhost",
                port=8000,
                temperature=0.1,
                top_p=0.9,
                model_name="dots_orc_1_5",
            )
            save_dir = f"../data/{file_name_without_ext}"
            os.makedirs(save_dir, exist_ok=True)
            save_file_name = f"{save_dir}/{file_name_without_ext}_{idx}.md"
            with open(save_file_name, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"第{idx + 1}页解析完成，保存路径: {save_file_name}")

            # pdf转jpg图片保存
            # with open(f"data/{file_name}_{idx}.jpg", "wb") as f:
            #     img_byte_arr = BytesIO()
            #     img.save(img_byte_arr, format="JPEG")
            #     img_byte_arr = img_byte_arr.getvalue()
            #     f.write(img_byte_arr)
    else:
        response = inference_with_vllm(
            images,
            prompt,
            ip="localhost",
            port=8000,
            temperature=0.1,
            top_p=0.9,
            model_name="dots_orc_1_5",
        )
        with open(f"data/{file_name_without_ext}.md", "w", encoding="utf-8") as f:
            f.write(response)
    print("write file(s) successfully")


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--file_path", "-f", type=str, required=True, help="需要解析的文件路径")
    args = arg_parser.parse_args()

    prompt = dict_promptmode_to_prompt["prompt_ocr"]

    file_path = args.file_path
    file_name, extension = os.path.splitext(file_path)

    assert extension in [".pdf", ".png", ".jpg", ".jpeg"]
    # 获取文件名
    name_without_ext = os.path.splitext(os.path.basename(file_path))[0]

    if extension == ".pdf":
        images = load_images_from_pdf(file_path)
        print(f"正在解析PDF文件({os.path.basename(file_path)})，共{len(images)}页")
    else:
        images = Image.open(file_path)
        print(f"正在解析图片({os.path.basename(file_path)})")
    convert_image_to_markdown(images, name_without_ext)


