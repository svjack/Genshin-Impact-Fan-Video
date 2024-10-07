import numpy as np
import os
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
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips, ImageClip, AudioFileClip
from moviepy.video import fx
from copy import deepcopy
import shutil
import zipfile
from tqdm import tqdm
#from IPython import display

# 配置外部变量
#HF_ENDPOINT = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')

# 加载数据集
character_image_ds = load_dataset("svjack/Genshin-Impact-Illustration")

# 下载数据集
if not os.path.exists("dialogue_video_merge_save_unique"):
    path = snapshot_download(
        repo_id="svjack/dialogue_video_merge_save_unique",
        repo_type="dataset",
        local_dir="dialogue_video_merge_save_unique",
        local_dir_use_symlinks=False
    )

# 读取文件内容
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# 加载对话数据
name_dialogue_dict = dict(pd.Series(
    list(pathlib.Path("dialogue_video_merge_save_unique/").glob("*.txt"))
).map(str).map(
    lambda x: (os.path.basename(x).replace("_dia.txt", ""), read_file(x))
).values.tolist())

# 解压文件
def extract_with_correct_encoding(zip_ref, extract_path):
    for file_info in zip_ref.infolist():
        try:
            file_info.filename = file_info.filename.encode("cp437").decode('gbk')
        except UnicodeDecodeError:
            pass
        zip_ref.extract(file_info, extract_path)

# 解压相关文件
def unzip_files(zip_name, folder_name):
    if not os.path.exists(folder_name):
        source_zip_path = os.path.join("dialogue_video_merge_save_unique", zip_name)
        destination_path = os.getcwd()
        shutil.copy(source_zip_path, destination_path)
        copied_zip_path = f"{folder_name}.zip"
        with zipfile.ZipFile(copied_zip_path, 'r') as zip_ref:
            extract_with_correct_encoding(zip_ref, destination_path)
        os.remove(copied_zip_path)

#unzip_files("原神角色背景图片（新）.zip", "原神角色背景图片（新）")
unzip_files("提瓦特音乐（人物）（新）.zip", "提瓦特音乐（人物）（新）")
unzip_files("d_audio.zip", "d_audio")

unzip_files("芭芭拉.zip", "芭芭拉")
unzip_files("安柏.zip", "安柏")
unzip_files("优菈.zip", "优菈")
unzip_files("香菱.zip", "香菱")
unzip_files("行秋.zip", "行秋")
unzip_files("重云.zip", "重云")

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

# 解析对话文本
def out_text_to_collection(out_text, name, meipo_name="珐露珊", traveler_name="荧", rp_name=True):
    pattern = re.compile(r'(\d+)\. \*\*(.*?)：(.*?)\*\* - (.*)')
    if rp_name:
        out_text = out_text.replace("男孩：", f"{name}：").replace("媒婆：", f"{meipo_name}：").replace("女孩：", f"{traveler_name}：")
        if name in girls:
            out_text = out_text.replace("他", "她").replace("小伙子", "大闺女").replace("小伙", "闺女").replace("男", "女")
    matches = pattern.findall(out_text)
    dialogues = []
    for match in matches:
        dialogues.append({
            'id': int(match[0]),
            'speaker': match[1],
            'content': match[2],
            'interpretation': match[3]
        })
    return dialogues

# 角色ID映射
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


# 生成缓存键
def generate_cache_key(text, id, format, lang, length, noise, noisew, segment_size, sdp_ratio):
    key = f"{text}_{id}_{format}_{lang}_{length}_{noise}_{noisew}_{segment_size}_{sdp_ratio}"
    return hashlib.sha256(key.encode()).hexdigest()

# 读取音频文件
def read_voice_bert_vits2_with_cache(text, id=0, format="wav", lang="auto", length=1, noise=0.667, noisew=0.8, segment_size=50, sdp_ratio=0.2, save_audio=True, save_path=None):
    cache_key = generate_cache_key(text, id, format, lang, length, noise, noisew, segment_size, sdp_ratio)
    cache_file = f"{cache_key}.{format}"
    cache_path = os.path.join(save_path, cache_file) if save_path else os.path.join(absolute_path, cache_file)
    assert os.path.exists(cache_path)
    return cache_path

# 解析对话文本并生成音频
def out_text_to_audio_read(out_text, audio_save_path, name, meipo_name="珐露珊", traveler_name="荧"):
    assert name in all_characters_in_settings
    assert meipo_name in all_characters_in_settings
    assert traveler_name in all_characters_in_settings
    dialogues = out_text_to_collection(out_text, name, meipo_name, traveler_name)
    req = []
    for d in tqdm(dialogues):
        speaker = d["speaker"]
        content = d["content"]
        assert speaker in all_characters_in_settings
        content_l = content.split("\\")
        d["content_l"] = content_l
        d["content_audio_l"] = []
        for c in content_l:
            audio_path = read_voice_bert_vits2_with_cache(c, save_path=audio_save_path, id=spk2id[f"{speaker}_ZH"])
            d["content_audio_l"].append(audio_path)
        req.append(d)
    return req

# 处理视频和音频
def process_video_audio(dl):
    audio_list = pd.DataFrame(dl)["content_audio_l"].explode().dropna().values.tolist()
    audio_list = list(map(lambda x: os.path.join(os.getcwd(), x), audio_list))
    dl_add = []
    for ele in tqdm(dl):
        d = deepcopy(ele)
        video_root_path = "output_videos/"
        d["content_mp4_l"] = pd.Series(d["content_audio_l"]).map(lambda x: x.split("/")[-1]).map(lambda x: x.replace(".wav", ".mp4")).map(lambda x: os.path.join(video_root_path, x)).values.tolist()
        assert pd.Series(d["content_mp4_l"]).map(os.path.exists).all()
        dl_add.append(d)
    return dl_add

# 处理视频剪辑
def process_dl_add(dl_add, main_character):
    skip_header_time = 10
    #background_clip = VideoFileClip('{}.mp4'.format(main_character)).subclip(skip_header_time)
    background_clip = VideoFileClip(
        os.path.join("原神剪辑视频" ,'{}.mp4'.format(main_character))
    ).subclip(skip_header_time)
    background_clip = background_clip.without_audio()
    first_frame = background_clip.get_frame(0)
    top, bottom, left, right = detect_black_edges(first_frame)
    background_clip = background_clip.crop(x1=left, y1=top, x2=right + 1, y2=bottom + 1)
    output_dir = 'merge_output_videos'
    os.makedirs(output_dir, exist_ok=True)
    new_data = []
    total_time = 0
    current_speaker = None
    position_left = False
    vertical_margin_ratio = 0.1
    horizontal_margin_ratio = 0.1
    foreground_towards_dict = {'七七': 'left',
 '丽莎': 'left',
 '久岐忍': 'left',
 '九条裟罗': 'left',
 '云堇': 'right',
 '五郎': 'left',
 '优菈': 'left',
 '八重神子': 'left',
 '凝光': 'right',
 '凯亚': 'left',
 '刻晴': 'left',
 '北斗': 'left',
 '卡维': 'left',
 '可莉': 'left',
 '坎蒂丝': 'right',
 '埃洛伊': 'right',
 '夏沃蕾': 'left',
 '夏洛蒂': 'right',
 '多莉': 'left',
 '夜兰': 'left',
 '妮露': 'left',
 '娜维娅': 'left',
 '安柏': 'left',
 '宵宫': 'left',
 '托马': 'right',
 '提纳里': 'left',
 '早柚': 'right',
 '林尼': 'left',
 '枫原万叶': 'left',
 '柯莱': 'left',
 '流浪者': 'left',
 '温迪': 'left',
 '烟绯': 'left',
 '珊瑚宫心海': 'right',
 '珐露珊': 'left',
 '班尼特': 'left',
 '琳妮特': 'right',
 '琴': 'right',
 '瑶瑶': 'left',
 '甘雨': 'left',
 '申鹤': 'right',
 '白术': 'left',
 '砂糖': 'right',
 '神里绫人': 'right',
 '神里绫华': 'left',
 '米卡': 'right',
 '纳西妲': 'left',
 '绮良良': 'left',
 '罗莎莉亚': 'left',
 '胡桃': 'left',
 '艾尔海森': 'left',
 '芙宁娜': 'left',
 '芭芭拉': 'left',
 '荒泷一斗': 'left',
 '莫娜': 'left',
 '莱依拉': 'left',
 '莱欧斯利': 'left',
 '菲米尼': 'left',
 '菲谢尔': 'right',
 '行秋': 'right',
 '诺艾尔': 'right',
 '赛诺': 'right',
 '辛焱': 'left',
 '达达利亚': 'left',
 '迪卢克': 'right',
 '迪奥娜': 'left',
 '迪希雅': 'right',
 '那维莱特': 'right',
 '重云': 'left',
 '钟离': 'left',
 '阿贝多': 'left',
 '雷泽': 'left',
 '雷电将军': 'right',
 '香菱': 'left',
 '魈': 'left',
 '鹿野院平藏': 'left'}
    for item in dl_add[:3000]:
        if current_speaker != item["speaker"]:
            current_speaker = item["speaker"]
            position_left = not position_left
        item, total_time = process_item(item, background_clip, total_time,
                                        position_left, vertical_margin_ratio,
                                        horizontal_margin_ratio, output_dir, foreground_towards_dict)
        new_data.append(item)
    return new_data

# 处理单个条目
def process_item(item, background_clip, total_time, position_left, vertical_margin_ratio,
                 horizontal_margin_ratio, output_dir, foreground_towards_dict, lag_time=0.0):
    new_content_mp4_l = []
    for i, mp4_path in enumerate(item['content_mp4_l']):
        fore_path = os.path.join(item["speaker"], f"{item['speaker']}---{mp4_path.split('/')[-1]}")
        foreground_clip = load_video_clip(fore_path)
        if foreground_clip is None:
            continue
        replace_audio_path = os.path.join("d_audio", mp4_path.split("/")[-1].replace(".mp4", ".wav"))
        assert os.path.exists(replace_audio_path)
        replace_audio_clip = AudioFileClip(replace_audio_path)
        foreground_clip = foreground_clip.set_audio(replace_audio_clip)
        start_time = total_time
        end_time = start_time + foreground_clip.duration + lag_time
        background_subclip = background_clip.subclip(start_time, end_time)
        foreground_clip, fg_position = resize_and_position_foreground(foreground_clip, background_subclip, position_left, vertical_margin_ratio, horizontal_margin_ratio, item["speaker"], foreground_towards_dict)
        dia_background = load_and_process_dia_background("dia_background.png", alpha_ratio=0.7)
        dia_background = dia_background.set_start(0).set_duration(foreground_clip.duration)
        bg_width, bg_height = background_subclip.size
        dia_width = bg_width * 0.98
        dia_height = dia_background.size[1] * (dia_width / dia_background.size[0])
        dia_position = ("center", bg_height - dia_height - bg_height * 0.02)
        dia_background = dia_background.resize((dia_width, dia_height)).set_position(dia_position)
        composite_clip = CompositeVideoClip([background_subclip, dia_background, foreground_clip.set_position(fg_position)])
        output_path = os.path.join(output_dir, f"composite_{item['id']}_{i}.mp4")
        composite_clip.write_videofile(output_path, codec='libx264')
        new_content_mp4_l.append(output_path)
        total_time += foreground_clip.duration + lag_time
    item['content_merge_mp4_l'] = new_content_mp4_l
    return item, total_time

# 检测黑边
def detect_black_edges(frame, threshold=10):
    frame = frame.astype(np.float32)
    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    gray = np.where(gray < threshold, 0, 255)
    top = 0
    while top < gray.shape[0] and np.all(gray[top, :] == 0):
        top += 1
    bottom = gray.shape[0] - 1
    while bottom > 0 and np.all(gray[bottom, :] == 0):
        bottom -= 1
    left = 0
    while left < gray.shape[1] and np.all(gray[:, left] == 0):
        left += 1
    right = gray.shape[1] - 1
    while right > 0 and np.all(gray[:, right] == 0):
        right -= 1
    return top, bottom, left, right

# 加载视频剪辑
def load_video_clip(path, has_mask=True):
    try:
        clip = VideoFileClip(path, has_mask=has_mask)
        clip = fx.all.mask_color(clip, color=(0, 0, 0), thr=10, s=5)
        return clip
    except Exception as e:
        print(f"Error loading video clip {path}: {e}")
        return None

# 加载并处理字幕背景图片
def load_and_process_dia_background(image_path, alpha_ratio=0.1):
    dia_background = ImageClip(image_path)
    dia_background = dia_background.set_opacity(alpha_ratio)
    return dia_background

# 调整前景视频的大小并设置位置
def resize_and_position_foreground(foreground_clip, background_clip, position_left, vertical_margin_ratio, horizontal_margin_ratio, speaker, foreground_towards_dict):
    bg_width, bg_height = background_clip.size
    fg_width, fg_height = foreground_clip.size
    scale_factor = min(bg_width / fg_width, bg_height / fg_height) / 1.5
    foreground_clip = foreground_clip.resize(scale_factor)
    vertical_margin = bg_height * vertical_margin_ratio
    horizontal_margin = bg_width * horizontal_margin_ratio
    fg_center_x = int((bg_width - fg_width) * 0.5)
    fg_center_y = int((bg_height - fg_height) * 0.1)
    if position_left:
        fg_position = (horizontal_margin, fg_center_y + vertical_margin)
        if foreground_towards_dict.get(speaker) == "left":
            foreground_clip = fx.all.mirror_x(foreground_clip)
    else:
        fg_position = (bg_width - fg_width - horizontal_margin, fg_center_y + vertical_margin)
        if foreground_towards_dict.get(speaker) == "right":
            foreground_clip = fx.all.mirror_x(foreground_clip)
    return foreground_clip, fg_position

# 生成最终视频
def generate_final_video(data, output_path, blank_duration_ms=0, fontsize=100, font="simhei.ttf", buttom_pos=200):
    def get_audio_duration(audio_path):
        audio = AudioSegment.from_file(audio_path)
        return len(audio)

    def format_time(milliseconds):
        td = timedelta(milliseconds=milliseconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{td.microseconds // 1000:03}"

    def generate_srt(content_l, audio_durations, interval_ms, speaker):
        srt_content = []
        start_time = 0
        for i, (text, duration) in enumerate(zip(content_l, audio_durations)):
            text_ = f"{speaker} >    {text}"
            end_time = start_time + duration
            srt_content.append(f"{i+1}")
            srt_content.append(f"{format_time(start_time)} --> {format_time(end_time)}")
            srt_content.append(text_)
            srt_content.append("")
            start_time = end_time + interval_ms
        return "\n".join(srt_content)

    def generate_video(video_paths, srt_content, audio_durations, output_path, blank_duration_ms=1000, fontsize=78, font="simhei.ttf", buttom_pos=10, fade_val=0.1):
        temp_dir = os.path.join(os.getcwd(), f"temp_{hashlib.md5(str(random.random()).encode()).hexdigest()}")
        os.makedirs(temp_dir, exist_ok=True)
        video_clips = [VideoFileClip(video_path) for video_path in video_paths]
        audio_clips = [video_clip.audio for video_clip in video_clips]

        video_resol_w = video_clip.size[0]

        audio_clips_with_fade = []
        for i, audio_clip in enumerate(audio_clips):
            if i > 0:
                audio_clip = audio_clip.audio_fadeout(fade_val)
            if i < len(audio_clips) - 1:
                audio_clip = audio_clip.audio_fadein(fade_val)
            audio_clips_with_fade.append(audio_clip)
        audio_concat = concatenate_audioclips(audio_clips_with_fade)
        video_clip = concatenate_videoclips(video_clips, method="compose")
        video_clip = video_clip.set_duration(audio_concat.duration)
        video_clip = video_clip.set_audio(audio_concat)
        subtitles = pysrt.from_string(srt_content)
        subtitle_clips = []
        current_blank_duration = 0
        for subtitle in subtitles:
            start_time = (subtitle.start.ordinal / 1000) + current_blank_duration
            end_time = (subtitle.end.ordinal / 1000) + current_blank_duration
            text = subtitle.text
            #### 默认 1280x720 下的合适 fontsize_input = 20
            fontsize_input = 20
            #### 2560 下
            fontsize_input = 40
            #### 1920 下
            fontsize_input = 30

            if video_resol_w > 2000:
                fontsize_input = 40
            else:
                fontsize_input = 30

            subtitle_clip = TextClip(text, fontsize=fontsize_input, color='white', font=font)
            ea_diff = end_time - start_time
            ea_diff = ea_diff * 0.8
            subtitle_clip = subtitle_clip.set_position(('center', 'bottom')).set_duration(ea_diff).set_start(start_time + ea_diff * 0.1)
            subtitle_clip = subtitle_clip.set_position(lambda t: (video_clip.w / 9.3, video_clip.h - int(video_clip.h * 0.15)))
            subtitle_clips.append(subtitle_clip)
            current_blank_duration += blank_duration_ms / 1000
        final_clip = CompositeVideoClip([video_clip] + subtitle_clips)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        shutil.rmtree(temp_dir)

    new_data = []
    for item in data:
        new_item = item.copy()
        new_item['audio_durations'] = []
        for audio_file in item['content_audio_l']:
            duration = get_audio_duration(audio_file)
            new_item['audio_durations'].append(duration)
        new_data.append(new_item)

    for item in new_data:
        srt_string = generate_srt(item['content_l'], item['audio_durations'], 0, item["speaker"])
        item['srt'] = srt_string

    video_paths = []
    for item in new_data:
        video_hash = hashlib.md5(str(random.random()).encode()).hexdigest()
        video_path = f"temp_video_{video_hash}.mp4"
        generate_video(item["content_merge_mp4_l"], item["srt"], item["audio_durations"], video_path, blank_duration_ms, fontsize, font, buttom_pos)
        video_paths.append(video_path)

    video_clips = [VideoFileClip(video_path) for video_path in video_paths]
    final_clip = concatenate_videoclips(video_clips, method="compose")
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    for video_path in video_paths:
        if os.path.exists(video_path):
            os.remove(video_path)

# 添加背景音乐
def add_background_music(video_path, background_music_path, output_path, music_volume=0.5, fade_duration=2):
    videoclip = VideoFileClip(video_path)
    soundtrack = AudioFileClip(background_music_path)
    if soundtrack.duration < videoclip.duration:
        soundtrack_looped = soundtrack.audio_loop(duration=videoclip.duration)
    else:
        soundtrack_looped = soundtrack.subclip(0, videoclip.duration)
    soundtrack_looped = soundtrack_looped.volumex(music_volume)
    soundtrack_looped = audio_fadein(soundtrack_looped, fade_duration)
    soundtrack_looped = audio_fadeout(soundtrack_looped, fade_duration)
    original_audio = videoclip.audio
    final_audio = CompositeAudioClip([original_audio, soundtrack_looped])
    videoclip.audio = final_audio
    videoclip.write_videofile(output_path, codec='libx264')

# 清理临时文件
def cleanup_temp_files():
    temp_dir = os.path.join(os.getcwd(), "temp_*")
    for temp_folder in glob.glob(temp_dir):
        shutil.rmtree(temp_folder)

demo_list = [
    "香菱",
    "芭芭拉",
]

# 主函数
def main():
    #from IPython import display
    for name in tqdm(name_dialogue_dict.keys()):
        if name not in demo_list:
            continue

        output_dir = 'merge_output_videos'
        os.makedirs(output_dir, exist_ok=True)

        audio_path = "d_audio"
        dialogue_text = name_dialogue_dict[name]
        dl = out_text_to_audio_read(dialogue_text, audio_path, name, relation_mapping[name]["媒婆"], relation_mapping[name]["女孩"])
        #display.clear_output(wait = True)

        dl_add = process_video_audio(dl)
        dl_add_out = process_dl_add(dl_add, name)
        #display.clear_output(wait = True)

        final_video_path = os.path.join(output_dir, f"{name}_composite.mp4")
        generate_final_video(
            list(map(lambda d: dict(filter(lambda t2: t2[0] in ["content_l", "content_audio_l", "content_merge_mp4_l", "speaker"], d.items())), dl_add_out[:30000])),
            final_video_path
        )
        #display.clear_output(wait = True)

        background_music_path = os.path.join("提瓦特音乐（人物）（新）", f"{name}.mp3")
        if os.path.exists(background_music_path):
            #final_output_path = os.path.join(output_dir, f"{name}动态.mp4")
            final_output_path = f"{name}动态.mp4"
            add_background_music(final_video_path, background_music_path, final_output_path)
            os.remove(final_video_path)  # 删除中间生成的视频文件
        #display.clear_output(wait = True)

        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        #display.clear_output(wait = True)

if __name__ == "__main__":
    main()
