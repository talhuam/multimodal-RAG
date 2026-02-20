from openai import OpenAI
import requests
import os
from io import BytesIO
import base64
from PIL import Image
import fitz


dict_promptmode_to_prompt = {
    # prompt_layout_all_en: parse all layout info in json format.
    "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""",

    # prompt_layout_only_en: layout detection
    "prompt_layout_only_en": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",

    # prompt_ocr: parse ocr text except the Page-header and Page-footer
    "prompt_ocr": """Extract the text content from this image.""",

    # prompt_grounding_ocr: extract text content in the given bounding box
    "prompt_grounding_ocr": """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",

    # prompt_web_parsing: parse all webpage layout info in json format.
    "prompt_web_parsing": """Parsing the layout info of this webpage image with format json:\n""",

    # prompt_scene_spotting: scene spotting
    "prompt_scene_spotting": """Detect and recognize the text in the image.""",

    # prompt_img2svg: generate the SVG code of the image
    "prompt_image_to_svg": """Please generate the SVG code based on the image.viewBox="0 0 {width} {height}\"""",

    # prompt_free_qa: general prompt
    "prompt_general": """ """,

    # "prompt_table_html": """Convert the table in this image to HTML.""",
    # "prompt_table_latex": """Convert the table in this image to LaTeX.""",
    # "prompt_formula_latex": """Convert the formula in this image to LaTeX.""",
}


def PILimage_to_base64(image, format='PNG'):
    buffered = BytesIO()
    image.save(buffered, format=format)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{base64_str}"


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


def fitz_doc_to_image(doc, target_dpi=200, origin_dpi=None) -> dict:
    """Convert fitz.Document to image, Then convert the image to numpy array.

    Args:
        doc (_type_): pymudoc page
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img': numpy array, 'width': width, 'height': height }
    """
    from PIL import Image
    mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
    pm = doc.get_pixmap(matrix=mat, alpha=False)

    if pm.width > 4500 or pm.height > 4500:
        mat = fitz.Matrix(72 / 72, 72 / 72)  # use fitz default dpi
        pm = doc.get_pixmap(matrix=mat, alpha=False)

    image = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    return image


def load_images_from_pdf(pdf_file, dpi=200, start_page_id=0, end_page_id=None) -> list:
    images = []
    with fitz.open(pdf_file) as doc:
        pdf_page_num = doc.page_count
        end_page_id = (
            end_page_id
            if end_page_id is not None and end_page_id >= 0
            else pdf_page_num - 1
        )
        if end_page_id > pdf_page_num - 1:
            print('end_page_id is out of range, use images length')
            end_page_id = pdf_page_num - 1

        for index in range(0, doc.page_count):
            if start_page_id <= index <= end_page_id:
                page = doc[index]
                img = fitz_doc_to_image(page, target_dpi=dpi)
                images.append(img)
    return images


def convert_image_to_markdown(images, file_name):
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
            with open(f"data/{file_name}_{idx}.md", "w", encoding="utf-8") as f:
                f.write(response)

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
        with open(f"data/{file_name}.md", "w", encoding="utf-8") as f:
            f.write(response)
    print("write file(s) successfully")


if __name__ == '__main__':
    prompt = dict_promptmode_to_prompt["prompt_ocr"]

    file_path = r"./data/demo_pdf1.pdf"
    file_name, extension = os.path.splitext(file_path)
    # 获取文件名
    name_without_ext = os.path.splitext(os.path.basename(file_path))[0]

    if extension == ".pdf":
        images = load_images_from_pdf(file_path)
    else:
        images = Image.open(file_path)
    convert_image_to_markdown(images, name_without_ext)


