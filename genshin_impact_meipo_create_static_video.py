# coding:utf-8

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import requests
import random
import string
import hashlib
import pathlib
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
import io
from moviepy.config import change_settings
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, CompositeAudioClip, concatenate_videoclips, concatenate_audioclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.tools.subtitles import SubtitlesClip
from pydub import AudioSegment
from datetime import timedelta
import pysrt
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
from copy import deepcopy
import shutil
import zipfile

character_image_ds = load_dataset("svjack/Genshin-Impact-Illustration")

if not os.path.exists("dialogue_feat_merge_save_unique"):
    path = snapshot_download(
        repo_id="svjack/dialogue_feat_merge_save_unique",
        repo_type="dataset",
        local_dir="dialogue_feat_merge_save_unique",
        local_dir_use_symlinks = False
    )

def r_func(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

name_dialogue_dict = dict(pd.Series(
    list(pathlib.Path("dialogue_feat_merge_save_unique/").glob("*.txt"))
).map(str).map(
    lambda x: (os.path.basename(x).replace("_dia.txt", ""), r_func(x))
).values.tolist())
#print(name_dialogue_dict["丽莎"])

def extract_with_correct_encoding(zip_ref, extract_path):
    for file_info in zip_ref.infolist():
        try:
            # 尝试使用utf-8解码文件名
            file_info.filename = file_info.filename.encode("cp437").decode('gbk')
        except UnicodeDecodeError:
            # 如果解码失败，保留原始文件名
            pass
        zip_ref.extract(file_info, extract_path)


if not os.path.exists("原神角色背景图片（新）"):
    source_zip_path = os.path.join("dialogue_feat_merge_save_unique", "原神角色背景图片（新）.zip")
    destination_path = os.getcwd()
    shutil.copy(source_zip_path, destination_path)
    copied_zip_path = "原神角色背景图片（新）.zip"
    with zipfile.ZipFile(copied_zip_path, 'r') as zip_ref:
        extract_with_correct_encoding(zip_ref, destination_path)
    os.remove(copied_zip_path)

if not os.path.exists("提瓦特音乐（人物）（新）"):
    source_zip_path = os.path.join("dialogue_feat_merge_save_unique", "提瓦特音乐（人物）（新）.zip")
    destination_path = os.getcwd()
    shutil.copy(source_zip_path, destination_path)
    copied_zip_path = "提瓦特音乐（人物）（新）.zip"
    with zipfile.ZipFile(copied_zip_path, 'r') as zip_ref:
        extract_with_correct_encoding(zip_ref, destination_path)
    os.remove(copied_zip_path)

if not os.path.exists("d_audio"):
    source_zip_path = os.path.join("dialogue_feat_merge_save_unique", "d_audio.zip")
    destination_path = os.getcwd()
    shutil.copy(source_zip_path, destination_path)
    copied_zip_path = "d_audio.zip"
    with zipfile.ZipFile(copied_zip_path, 'r') as zip_ref:
        extract_with_correct_encoding(zip_ref, destination_path)
    os.remove(copied_zip_path)


#### windows in windows (make sure install magick)
change_settings({"IMAGEMAGICK_BINARY": "D:/software_ins/ImageMagick-7.1.1-Q16-HDRI/magick.exe"})

#### 定义3元组图谱
relation_mapping = {'丽莎': {'媒婆': '可莉', '女孩': '雷泽'},
 '行秋': {'媒婆': '申鹤', '女孩': '重云'},
 '优菈': {'媒婆': '琴', '女孩': '安柏'},
 '魈': {'媒婆': '胡桃', '女孩': '钟离'},
 '五郎': {'媒婆': '珊瑚宫心海', '女孩': '八重神子'},
 '钟离': {'媒婆': '北斗', '女孩': '凝光'},
 '温迪': {'媒婆': '雷电将军', '女孩': '钟离'},
 '菲谢尔': {'媒婆': '丽莎', '女孩': '班尼特'},
 '诺艾尔': {'媒婆': '罗莎莉亚', '女孩': '芭芭拉'},
 '云堇': {'媒婆': '凝光', '女孩': '申鹤'},
 '班尼特': {'媒婆': '安柏', '女孩': '雷泽'},
 '安柏': {'媒婆': '珐露珊', '女孩': '柯莱'},
 '莫娜': {'媒婆': '温迪', '女孩': '菲谢尔'},
 '柯莱': {'媒婆': '妮露', '女孩': '砂糖'},
 '迪卢克': {'媒婆': '迪奥娜', '女孩': '凯亚'},
 '绮良良': {'媒婆': '夏洛蒂', '女孩': '娜维娅'},
 '阿贝多': {'媒婆': '砂糖', '女孩': '丽莎'},
 '卡维': {'媒婆': '纳西妲', '女孩': '艾尔海森'},
 '提纳里': {'媒婆': '多莉', '女孩': '赛诺'},
 '夏沃蕾': {'媒婆': '娜维娅', '女孩': '莱欧斯利'},
 '鹿野院平藏': {'媒婆': '九条裟罗', '女孩': '枫原万叶'},
 '辛焱': {'媒婆': '香菱', '女孩': '云堇'},
 '八重神子': {'媒婆': '神里绫华', '女孩': '雷电将军'},
 '凝光': {'媒婆': '刻晴', '女孩': '北斗'},
 '九条裟罗': {'媒婆': '久岐忍', '女孩': '荒泷一斗'},
 '米卡': {'媒婆': '优菈', '女孩': '诺艾尔'},
 '刻晴': {'媒婆': '甘雨', '女孩': '行秋'},
 '罗莎莉亚': {'媒婆': '芭芭拉', '女孩': '阿贝多'},
 '甘雨': {'媒婆': '刻晴', '女孩': '申鹤'},
 '砂糖': {'媒婆': '诺艾尔', '女孩': '雷泽'},
 '坎蒂丝': {'媒婆': '多莉', '女孩': '迪希雅'},
 '枫原万叶': {'媒婆': '北斗', '女孩': '流浪者'},
 '雷电将军': {'媒婆': '宵宫', '女孩': '八重神子'},
 '白术': {'媒婆': '七七', '女孩': '胡桃'},
 '娜维娅': {'媒婆': '夏沃蕾', '女孩': '芙宁娜'},
 '芙宁娜': {'媒婆': '琳妮特', '女孩': '菲米尼'},
 '莱欧斯利': {'媒婆': '芙宁娜', '女孩': '那维莱特'},
 '迪奥娜': {'媒婆': '可莉', '女孩': '迪卢克'},
 '夜兰': {'媒婆': '甘雨', '女孩': '凝光'},
 '久岐忍': {'媒婆': '烟绯', '女孩': '鹿野院平藏'},
 '菲米尼': {'媒婆': '琳妮特', '女孩': '林尼'},
 '可莉': {'媒婆': '迪奥娜', '女孩': '雷泽'},
 '流浪者': {'媒婆': '纳西妲', '女孩': '妮露'},
 '多莉': {'媒婆': '珐露珊', '女孩': '卡维'},
 '凯亚': {'媒婆': '琴', '女孩': '迪卢克'},
 '琴': {'媒婆': '丽莎', '女孩': '芭芭拉'},
 '琳妮特': {'媒婆': '芙宁娜', '女孩': '绮良良'},
 '荒泷一斗': {'媒婆': '八重神子', '女孩': '久岐忍'},
 '神里绫人': {'媒婆': '雷电将军', '女孩': '那维莱特'},
 '夏洛蒂': {'媒婆': '夏沃蕾', '女孩': '娜维娅'},
 '雷泽': {'媒婆': '可莉', '女孩': '班尼特'},
 '芭芭拉': {'媒婆': '安柏', '女孩': '优菈'},
 '珊瑚宫心海': {'媒婆': '雷电将军', '女孩': '八重神子'},
 '妮露': {'媒婆': '纳西妲', '女孩': '流浪者'},
 '七七': {'媒婆': '香菱', '女孩': '瑶瑶'},
 '香菱': {'媒婆': '行秋', '女孩': '重云'},
 '珐露珊': {'媒婆': '柯莱', '女孩': '莱依拉'},
 '赛诺': {'媒婆': '妮露', '女孩': '柯莱'},
 '神里绫华': {'媒婆': '宵宫', '女孩': '枫原万叶'},
 '申鹤': {'媒婆': '云堇', '女孩': '魈'},
 '瑶瑶': {'媒婆': '七七', '女孩': '香菱'},
 '达达利亚': {'媒婆': '流浪者', '女孩': '钟离'},
 '早柚': {'媒婆': '神里绫华', '女孩': '托马'},
 '北斗': {'媒婆': '珊瑚宫心海', '女孩': '枫原万叶'},
 '重云': {'媒婆': '香菱', '女孩': '行秋'},
 '林尼': {'媒婆': '琳妮特', '女孩': '芙宁娜'},
 '埃洛伊': {'媒婆': '丽莎', '女孩': '琴'},
 '托马': {'媒婆': '神里绫华', '女孩': '枫原万叶'},
 '纳西妲': {'媒婆': '芙宁娜', '女孩': '温迪'},
 '烟绯': {'媒婆': '凝光', '女孩': '夜兰'},
 '那维莱特': {'媒婆': '芙宁娜', '女孩': '神里绫人'},
 '迪希雅': {'媒婆': '坎蒂丝', '女孩': '赛诺'},
 '宵宫': {'媒婆': '雷电将军', '女孩': '神里绫华'},
 '胡桃': {'媒婆': '可莉', '女孩': '七七'},
 '艾尔海森': {'媒婆': '纳西妲', '女孩': '卡维'},
 '莱依拉': {'媒婆': '多莉', '女孩': '迪希雅'}}

all_characters_in_settings = ['丽莎', '行秋', '优菈', '魈', '五郎', '钟离', '温迪',
'菲谢尔', '诺艾尔', '云堇',
 '班尼特', '安柏', '莫娜', '柯莱', '迪卢克', '绮良良', '阿贝多', '卡维', '提纳里',
 '夏沃蕾', '鹿野院平藏', '辛焱', '八重神子', '凝光', '九条裟罗', '米卡', '刻晴',
 '罗莎莉亚', '甘雨', '砂糖', '坎蒂丝', '枫原万叶', '雷电将军', '白术', '娜维娅',
 '芙宁娜', '莱欧斯利', '迪奥娜', '夜兰', '久岐忍', '菲米尼', '可莉', '流浪者',
 '多莉', '凯亚', '琴', '琳妮特', '荒泷一斗', '神里绫人', '夏洛蒂', '雷泽', '芭芭拉',
 '珊瑚宫心海', '妮露', '七七', '香菱', '珐露珊', '赛诺', '神里绫华', '申鹤', '瑶瑶',
 '达达利亚', '早柚', '北斗', '重云', '林尼', '埃洛伊', '托马', '纳西妲', '烟绯',
 '那维莱特', '迪希雅', '宵宫', '胡桃', '艾尔海森', '莱依拉']

all_characters_in_settings = all_characters_in_settings + ["空", "荧"]

girls = ['丽莎', '优菈',
 '菲谢尔', '诺艾尔', '云堇',
   '安柏', '莫娜', '柯莱', '绮良良',
  '夏沃蕾',  '辛焱', '八重神子', '凝光', '九条裟罗', '刻晴',
  '罗莎莉亚', '甘雨', '砂糖', '坎蒂丝', '雷电将军', '娜维娅',
  '芙宁娜', '迪奥娜', '夜兰', '久岐忍', '可莉',
  '多莉', '琴', '琳妮特', '夏洛蒂', '芭芭拉',
  '珊瑚宫心海', '妮露', '七七', '香菱', '珐露珊','神里绫华', '申鹤', '瑶瑶',
  '早柚', '北斗', '埃洛伊', '纳西妲', '烟绯',
  '迪希雅', '宵宫', '胡桃', '莱依拉']

girls = girls + ["荧"]

def out_text_to_collection(out_text, name, meipo_name = "珐露珊", traveler_name = "荧", rp_name = True):
    import re
    pattern = re.compile(r'(\d+)\. \*\*(.*?)：(.*?)\*\* - (.*)')

    if rp_name:

        #### name + girls process
        #out_text = out_text.replace("男孩", name).replace("媒婆", meipo_name).replace("女孩", traveler_name)
        ##### 男孩：
        out_text = out_text.replace("男孩：", "{}：".format(name)).replace("媒婆：", "{}：".format(meipo_name)).replace("女孩：", "{}：".format(traveler_name))
        if name in girls:
            out_text = out_text.replace("他", "她").replace("小伙子", "大闺女").replace("小伙", "闺女").replace("男", "女")

    # 解析字符串
    matches = pattern.findall(out_text)
    # 转换为列表字典对象
    dialogues = []
    for match in matches:
        dialogue = {
                'id': int(match[0]),
                'speaker': match[1],
                'content': match[2],
                'interpretation': match[3]
        }
        dialogues.append(dialogue)
    return dialogues


spk2id = {'纳西妲_ZH': 1,
 '凯亚_ZH': 2,
 '温迪_ZH': 3,
 '荒泷一斗_ZH': 4,
 '娜维娅_ZH': 5,
 '阿贝多_ZH': 6,
 '钟离_ZH': 7,
 '枫原万叶_ZH': 8,
 '那维莱特_ZH': 9,
 '艾尔海森_ZH': 10,
 '八重神子_ZH': 11,
 '宵宫_ZH': 12,
 '芙宁娜_ZH': 13,
 '迪希雅_ZH': 14,
 '提纳里_ZH': 15,
 '莱依拉_ZH': 16,
 '卡维_ZH': 17,
 '诺艾尔_ZH': 18,
 '赛诺_ZH': 19,
 '林尼_ZH': 20,
 '莫娜_ZH': 21,
 '托马_ZH': 22,
 '神里绫华_ZH': 23,
 '凝光_ZH': 24,
 '北斗_ZH': 25,
 '可莉_ZH': 26,
 '柯莱_ZH': 27,
 '迪奥娜_ZH': 28,
 '莱欧斯利_ZH': 29,
 '芭芭拉_ZH': 30,
 '雷电将军_ZH': 31,
 '珊瑚宫心海_ZH': 32,
 '魈_ZH': 33,
 '五郎_ZH': 34,
 '胡桃_ZH': 35,
 '鹿野院平藏_ZH': 36,
 '安柏_ZH': 37,
 '琴_ZH': 38,
 '重云_ZH': 39,
 '达达利亚_ZH': 40,
 '班尼特_ZH': 41,
 '夜兰_ZH': 42,
 '丽莎_ZH': 43,
 '香菱_ZH': 44,
 '妮露_ZH': 45,
 '刻晴_ZH': 46,
 '珐露珊_ZH': 47,
 '烟绯_ZH': 48,
 '辛焱_ZH': 49,
 '早柚_ZH': 50,
 '迪卢克_ZH': 51,
 '砂糖_ZH': 52,
 '云堇_ZH': 53,
 '久岐忍_ZH': 54,
 '神里绫人_ZH': 55,
 '优菈_ZH': 56,
 '甘雨_ZH': 57,
 '夏洛蒂_ZH': 58,
 '流浪者_ZH': 59,
 '行秋_ZH': 60,
 '夏沃蕾_ZH': 61,
 '白术_ZH': 64,
 '菲谢尔_ZH': 65,
 '申鹤_ZH': 66,
 '九条裟罗_ZH': 67,
 '雷泽_ZH': 68,
 '荧_ZH': 69,
 '空_ZH': 70,
 '菲米尼_ZH': 72,
 '多莉_ZH': 73,
 '琳妮特_ZH': 75,
 '米卡_ZH': 77,
 '坎蒂丝_ZH': 78,
 '罗莎莉亚_ZH': 80,
 '瑶瑶_ZH': 85,
 '绮良良_ZH': 86,
 '七七_ZH': 87,
 '埃洛伊_ZH': 110}


def generate_cache_key(text, id, format, lang, length, noise, noisew, segment_size, sdp_ratio):
    # 生成一个唯一的缓存键
    key = f"{text}_{id}_{format}_{lang}_{length}_{noise}_{noisew}_{segment_size}_{sdp_ratio}"
    return hashlib.sha256(key.encode()).hexdigest()

def read_voice_bert_vits2_with_cache(text, id=0, format="wav", lang="auto", length=1, noise=0.667, noisew=0.8, segment_size=50,
                     sdp_ratio=0.2, save_audio=True, save_path=None):
    # 生成缓存键
    cache_key = generate_cache_key(text, id, format, lang, length, noise, noisew, segment_size, sdp_ratio)
    cache_file = f"{cache_key}.{format}"

    if save_path is not None:
        cache_path = os.path.join(save_path, cache_file)
    else:
        cache_path = os.path.join(absolute_path, cache_file)

    # 检查缓存文件是否存在
    assert os.path.exists(cache_path)
    ###print(f"Cache hit: {cache_path}")
    return cache_path

def out_text_to_audio_read(out_text, audio_save_path, name, meipo_name = "珐露珊", traveler_name = "荧"):
    assert name in all_characters_in_settings
    assert meipo_name in all_characters_in_settings
    assert traveler_name in all_characters_in_settings
    dialogues = out_text_to_collection(out_text, name, meipo_name, traveler_name)
    from tqdm import tqdm
    req = []
    for d in tqdm(dialogues):
        speaker = d["speaker"]
        content = d["content"]
        assert speaker in all_characters_in_settings
        content_l = content.split("\\")
        d["content_l"] = content_l
        d["content_audio_l"] = []
        for c in content_l:
            audio_path = read_voice_bert_vits2_with_cache(
                c, save_path=audio_save_path,
                id = spk2id["{}_ZH".format(speaker)],
                )
            d["content_audio_l"].append(audio_path)
        req.append(d)
    return req


def bytes_to_image(image_bytes):
    """
    将图片的 bytes 数据转换为 Pillow 的 Image 对象。

    参数:
    image_bytes (bytes): 包含图片数据的 bytes 对象。

    返回:
    Image: 转换后的 Pillow Image 对象。
    """
    # 使用 io.BytesIO 将 bytes 数据转换为文件对象
    image_file = io.BytesIO(image_bytes)

    # 使用 Pillow 的 Image.open 方法打开文件对象，并返回 Image 对象
    image = Image.open(image_file)

    return image

def resize_image_by_factor(image, factor):
    # 获取原始图像的尺寸
    width, height = image.size

    # 计算新的尺寸
    new_width = int(width * factor)
    new_height = int(height * factor)

    # 调整图像大小
    resized_image = image.resize((new_width, new_height))

    return resized_image

from PIL import Image, ImageDraw
import numpy as np

#### 运算返回拷贝 原图不变
def apply_circular_mask(image, radius_ratio=0.9):
    """
    将图像的中心部分保留为圆形，圆形之外的部分变透明。

    :param image: 输入图像，PIL Image对象
    :param radius_ratio: 圆形半径的比例系数，范围为0到1，0表示没有圆形，1表示整个图像都是圆形
    :return: 应用圆形蒙版后的图像，PIL Image对象
    """

    # 获取图像的尺寸
    width, height = image.size

    # 计算圆形的半径
    radius = min(width, height) * radius_ratio / 2

    # 创建一个与图像大小相同的蒙版
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # 创建一个圆形蒙版
    for i in range(width):
        for j in range(height):
            distance = np.sqrt((i - width / 2) ** 2 + (j - height / 2) ** 2)
            if distance <= radius:
                alpha = 255  # 不透明
            else:
                alpha = 0  # 透明
            draw.point((i, j), fill=int(alpha))

    # 创建一个新的图像对象，并应用蒙版
    result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    result.paste(image, (0, 0), mask)

    return result

def overlay_image(background_img, foreground_img, position='left', vertical_position='middle', resize_ratio=0.5):
    """
    将人物图盖在背景图上，并保持透明图层不变。

    :param background_img: 背景图，PIL Image对象
    :param foreground_img: 人物图，PIL Image对象
    :param position: 人物图的水平位置，可以是'left'或'right'
    :param vertical_position: 人物图的垂直位置，可以是'top', 'middle', 或 'bottom'
    :param resize_ratio: 人物图相对于背景图的缩放比例，默认是0.5
    :return: 合成后的图像，PIL Image对象
    """

    # 获取背景图的尺寸
    bg_width, bg_height = background_img.size

    # 计算人物图的缩放尺寸
    fg_width, fg_height = foreground_img.size
    new_fg_width = int(bg_width * resize_ratio)
    new_fg_height = int(fg_height * (new_fg_width / fg_width))

    # 确保人物图不会超过背景图中间的部分
    if new_fg_height > bg_height / 2:
        new_fg_height = int(bg_height / 2)
        new_fg_width = int(fg_width * (new_fg_height / fg_height))

    # 调整人物图的大小
    foreground_img = foreground_img.resize((new_fg_width, new_fg_height))

    # 确保人物图具有透明通道（RGBA格式）
    if foreground_img.mode != 'RGBA':
        foreground_img = foreground_img.convert('RGBA')

    # 计算人物图在背景图上的水平位置
    if position == 'left':
        x_offset = 0
    elif position == 'right':
        x_offset = bg_width - new_fg_width
    else:
        raise ValueError("position参数必须是'left'或'right'")

    # 计算人物图在背景图上的垂直位置
    if vertical_position == 'top':
        y_offset = 0
    elif vertical_position == 'middle':
        y_offset = (bg_height - new_fg_height) // 2
    elif vertical_position == 'bottom':
        y_offset = bg_height - new_fg_height
    else:
        raise ValueError("vertical_position参数必须是'top', 'middle', 或 'bottom'")

    # 将人物图粘贴到背景图上，并保持透明度
    background_img.paste(foreground_img, (x_offset, y_offset), foreground_img)

    return background_img

character_image_df = character_image_ds["train"].to_pandas()
character_image_df["image"] = character_image_df["image"].map(lambda x: bytes_to_image(x["bytes"]))
character_image_df["h"] = character_image_df["image"].map(lambda x: x.height)
character_image_df["w"] = character_image_df["image"].map(lambda x: x.width)
character_image_df[["h", "w"]].describe()

#### take times for 86 resize
name_im_cty_dict = dict(character_image_df.apply(
    lambda x: (
        x["name"], {
            "country": x["country"],
            "image": apply_circular_mask(x["image"], 0.9)
        }
    ), axis = 1
).values.tolist())


def add_position(data_list, name_im_cty_dict, background_img, initial_position="left"):
    """
    为列表中的每个元素添加一个 `position` 值，值从 `["left", "right"]` 中选取，随着 `speaker` 切换而变化。
    同时，调用 `overlay_image` 函数生成 `overlay_image` 字段。

    :param data_list: 包含字典元素的列表
    :param name_im_cty_dict: 包含 speaker 名称和对应图像的字典
    :param background_img: 背景图像，PIL Image对象
    :param initial_position: 初始的 `position` 值，可以是 "left" 或 "right"
    :return: 添加了 `position` 和 `overlay_image` 值的列表的深拷贝
    """
    # 创建一个深拷贝以避免修改原始数据
    result_list = deepcopy(data_list)

    # 初始化当前的 position 值
    current_position = initial_position

    # 遍历列表中的每个元素
    for i in range(len(result_list)):
        # 添加 position 值
        result_list[i]['position'] = current_position

        # 获取当前 speaker 对应的图像
        if result_list[i]["speaker"] == "荧":
            foreground_img = name_im_cty_dict["芙宁娜"]["image"]
        else:
            foreground_img = name_im_cty_dict[result_list[i]["speaker"]]["image"]

        # 调用 overlay_image 函数生成 overlay_image
        result_list[i]['image'] = overlay_image(background_img.copy(), foreground_img.copy(), position=current_position)

        # 如果当前元素的 speaker 与下一个元素的 speaker 不同，切换 position 值
        if i < len(result_list) - 1 and result_list[i]['speaker'] != result_list[i + 1]['speaker']:
            current_position = "right" if current_position == "left" else "left"

    return result_list


#### originl step size == 4
def adjust_font_size_by_length(text, base_size=100, step_size=5, step_length=4, min_size=20):
    # 计算字符个数
    char_count = len(text)

    # 计算倍数因子
    factor = (char_count // step_length) + 1

    # 根据倍数因子调整字体大小
    font_size = base_size - (factor - 1) * step_size

    # 确保字体大小不小于最小值
    font_size = int(max(font_size, min_size))

    return font_size

def generate_final_video(data, output_path, blank_duration_ms=1000, fontsize=100, font="simhei.ttf", buttom_pos=200):
    """
    生成最终视频并删除所有临时文件
    :param data: 包含音频、字幕和图片信息的字典列表
    :param output_path: 最终输出视频的路径
    :param blank_duration_ms: 音频之间的留白时长（毫秒）
    :param fontsize: 字幕字体大小
    :param font: 字幕字体
    :param buttom_pos: 字幕距离底部的位置
    """
    def get_audio_duration(audio_path):
        """获取音频文件的时长（毫秒）"""
        audio = AudioSegment.from_wav(audio_path)
        return len(audio)  # 返回时长（毫秒）

    def format_time(milliseconds):
        """将毫秒格式化为 SRT 时间格式"""
        td = timedelta(milliseconds=milliseconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{td.microseconds // 1000:03}"

    def generate_srt(content_l, audio_durations, interval_ms=0):
        """生成 SRT 字符串"""
        srt_content = []
        start_time = 0
        for i, (text, duration) in enumerate(zip(content_l, audio_durations)):
            end_time = start_time + duration
            srt_content.append(f"{i+1}")
            srt_content.append(f"{format_time(start_time)} --> {format_time(end_time)}")
            srt_content.append(text)
            srt_content.append("")
            start_time = end_time + interval_ms
        return "\n".join(srt_content)

    def generate_video(image, audio_paths, srt_content, audio_durations, output_path, blank_duration_ms=1000,
                       fontsize=78, font="simhei.ttf", buttom_pos=50,
                        position = "left"):
        """根据图片、音频和字幕生成视频"""
        # 创建临时目录
        temp_dir = os.path.join(os.getcwd(), f"temp_{hashlib.md5(str(random.random()).encode()).hexdigest()}")
        os.makedirs(temp_dir, exist_ok=True)

        # 保存 PIL Image 对象到临时文件
        #### must be jpg
        image_path = os.path.join(temp_dir, "temp_image.jpg")
        image.copy().save(image_path)

        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件 {image_path} 不存在")
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件 {audio_path} 不存在")

        # 创建视频背景
        video_clip = VideoFileClip(image_path)

        # 加载音频
        audio_clips = [AudioFileClip(audio_path) for audio_path in audio_paths]

        # 在音频之间加入留白
        audio_clips_with_blank = []

        for i, audio_clip in enumerate(audio_clips):
            if i > 0:
                silence_segment = AudioSegment.silent(duration=blank_duration_ms)
                silence_temp_file = os.path.join(temp_dir, f"silence_{i}.wav")
                silence_segment.export(silence_temp_file, format="wav")
                silence_clip = AudioFileClip(silence_temp_file)
                audio_clips_with_blank.append(silence_clip)
            audio_clips_with_blank.append(audio_clip)

        # 在视频尾部加入音频留白
        silence_segment_end = AudioSegment.silent(duration=blank_duration_ms)
        silence_temp_file_end = os.path.join(temp_dir, f"silence_end.wav")
        silence_segment_end.export(silence_temp_file_end, format="wav")
        silence_clip_end = AudioFileClip(silence_temp_file_end)
        audio_clips_with_blank.append(silence_clip_end)

        # 连接所有音频剪辑
        audio_concat = concatenate_audioclips(audio_clips_with_blank)

        # 设置视频的持续时间与音频一致
        video_clip = video_clip.set_duration(audio_concat.duration)

        # 添加音频到视频
        video_clip = video_clip.set_audio(audio_concat)

        # 解析字幕字符串
        subtitles = pysrt.from_string(srt_content)

        # 创建一个空列表来存储字幕剪辑
        subtitle_clips = []

        # 计算字幕的开始时间和结束时间，考虑留白偏移
        current_blank_duration = 0
        for subtitle in subtitles:
            start_time = (subtitle.start.ordinal / 1000) + current_blank_duration  # pysrt 的时间戳是毫秒
            end_time = (subtitle.end.ordinal / 1000) + current_blank_duration
            text = subtitle.text

            fontsize_input = adjust_font_size_by_length(text)

            # 创建字幕剪辑
            #subtitle_clip = TextClip(text, fontsize=fontsize, color='white', font=font)
            subtitle_clip = TextClip(text, fontsize=fontsize_input, color='white', font=font)

            '''
            margin = int((video_clip.w - subtitle_clip.w) / 4)
            if position == "left":
                subtitle_clip = subtitle_clip.set_position(('right', 'center')).set_duration(
                    end_time - start_time
                ).set_start(start_time)
                #### video_clip.w - subtitle_clip.w - margin
                subtitle_clip = subtitle_clip.set_position(lambda t: (video_clip.w - subtitle_clip.w - margin, "center"))
            else:
                subtitle_clip = subtitle_clip.set_position(('left', 'center')).set_duration(
                    end_time - start_time
                ).set_start(start_time)
                subtitle_clip = subtitle_clip.set_position(lambda t: (margin, "center"))
            '''
            subtitle_clip = subtitle_clip.set_position(('center', 'bottom')).set_duration(
                    end_time - start_time
            ).set_start(start_time)
            subtitle_clip = subtitle_clip.set_position(lambda t: ('center', video_clip.h - subtitle_clip.h - buttom_pos))

            subtitle_clips.append(subtitle_clip)

            # 更新当前留白时长
            current_blank_duration += blank_duration_ms / 1000

        # 将字幕剪辑和视频剪辑合并
        final_clip = CompositeVideoClip([video_clip] + subtitle_clips)

        # 导出最终视频
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # 清除临时文件
        shutil.rmtree(temp_dir)

    # 为每个音频文件添加时长信息（毫秒）
    new_data = []
    for item in data:
        new_item = item.copy()
        new_item['audio_durations'] = []
        for audio_file in item['content_audio_l']:
            duration = get_audio_duration(audio_file)
            new_item['audio_durations'].append(duration)
        new_data.append(new_item)

    # 为每个字典元素添加 SRT 字符串
    for item in new_data:
        srt_string = generate_srt(item['content_l'], item['audio_durations'], 0)
        item['srt'] = srt_string

    # 生成并连接所有视频片段
    video_paths = []
    for item in new_data:
        # 生成随机哈希值作为视频文件名
        video_hash = hashlib.md5(str(random.random()).encode()).hexdigest()
        video_path = f"temp_video_{video_hash}.mp4"
        generate_video(
            item["image"],  # 使用 overlay_image 作为图片
            item["content_audio_l"],
            item["srt"],
            item["audio_durations"],
            video_path,
            blank_duration_ms,
            fontsize,
            font,
            buttom_pos,
            item["position"]
        )
        video_paths.append(video_path)

    # 加载所有生成的视频片段
    video_clips = [VideoFileClip(video_path) for video_path in video_paths]

    # 连接所有视频片段
    final_clip = concatenate_videoclips(video_clips, method="compose")

    # 导出最终视频
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # 删除所有临时生成的视频文件
    for video_path in video_paths:
        if os.path.exists(video_path):
            os.remove(video_path)

#### 处理音频比视频短的循环问题
def add_background_music(video_path, background_music_path, output_path, music_volume=0.5, fade_duration=2):
    """
    为视频添加背景音乐，并循环播放背景音乐直到与视频长度一致。

    :param video_path: 输入视频文件的路径
    :param background_music_path: 背景音乐文件的路径
    :param output_path: 输出视频文件的路径
    :param music_volume: 背景音乐的音量，范围为0到1，默认为0.5
    :param fade_duration: 音频淡入淡出的时长，单位为秒，默认为2秒
    """
    # 加载视频文件
    videoclip = VideoFileClip(video_path)

    # 加载背景音乐文件
    soundtrack = AudioFileClip(background_music_path)

    # 如果背景音乐比视频短，循环播放背景音乐直到与视频长度一致
    if soundtrack.duration < videoclip.duration:
        soundtrack_looped = soundtrack.audio_loop(duration=videoclip.duration)
    else:
        # 如果背景音乐比视频长，截取背景音乐的片段以匹配视频的长度
        soundtrack_looped = soundtrack.subclip(0, videoclip.duration)

    # 调整背景音乐的音量
    soundtrack_looped = soundtrack_looped.volumex(music_volume)

    # 在视频开头添加音频淡入效果
    soundtrack_looped = audio_fadein(soundtrack_looped, fade_duration)

    # 在视频结尾添加音频淡出效果
    soundtrack_looped = audio_fadeout(soundtrack_looped, fade_duration)

    # 获取视频的原始音频
    original_audio = videoclip.audio

    # 合并背景音乐和原始音频
    final_audio = CompositeAudioClip([original_audio, soundtrack_looped])

    # 将最终音频应用到视频
    videoclip.audio = final_audio

    # 保存最终视频
    videoclip.write_videofile(output_path, codec='libx264')

if __name__ == "__main__":
    #from IPython.display import clear_output
    from tqdm import tqdm
    audio_path = "d_audio"
    ### os.makedirs(audio_path)

    for name in tqdm(name_dialogue_dict.keys()):
        print("start: ", name)

        #### 读取LLM生成内容
        print("read text")
        dialogue_text = name_dialogue_dict[name]
        #### text data and audio data prepare
        dl = out_text_to_audio_read(dialogue_text, audio_path, name, relation_mapping[name]["媒婆"], relation_mapping[name]["女孩"])

        #### 原神角色背景图片
        #### 加入角色图片、背景图片，确定所有帧信息
        print("add image to background image")
        bk_img = Image.open(os.path.join("原神角色背景图片（新）", "{}.png".format(name))).copy()
        dl_pos = add_position(dl, name_im_cty_dict, bk_img)

        #### 合并上面所有元素 生成 无背景音乐视频
        print("merge video")
        generate_final_video(dl_pos, "{}(无背景音乐).mp4".format(name))

        #### 加入背景音乐
        #### 提瓦特音乐（人物）.zip
        print("add background music")
        add_background_music("{}(无背景音乐).mp4".format(name),
            os.path.join("提瓦特音乐（人物）（新）" ,"{}.mp3".format(name)) ,
            "{}.mp4".format(name),)

        #clear_output(wait = True)
        #break
