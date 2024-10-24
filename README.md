# Genshin-Impact-Fan-Video

## 项目介绍

**Genshin-Impact-Fan-Video** 是一个《原神》AI驱动的角色视频项目。通过结合大型语言模型（LLM）API生成角色互动文案，利用VITS技术进行语音合成，并结合先进的文生图及视频合成技术，创造了一系列游戏角色之间有趣场景。最终产出为短视频，旨在为玩家带来欢乐和娱乐。<br/>
<br/>
使用的素材的收集、生成和处理合成全部为个人完成，每个部分使用的技术均为开源非商业化技术。

## 项目特点

- **AI驱动的文案生成**：利用LLM API结合RAG技术，自动生成角色之间的互动对话。
- **VITS语音合成**：通过VITS技术，为每个角色生成逼真的语音，增强视频的沉浸感。
- **StableDiffusion立绘绘画**: 使用SD文生图模型为每个角色绘画一些，用于配合生成动态口型。
- **LivePortrait立绘动态化**: 使用liveportrait技术使角色张嘴。
- **视频合成技术**：结合视频合成技术，将角色互动、语音和背景音乐无缝融合，创造出高质量的短视频。
- **视频高清化技术**: 上传Bilibili的终稿，使用了视频高清化技术扩增了视频的分辨率（x2或x4）。

## 技术栈

- **LLM API**：用于生成角色互动文案。
- **VITS**：用于语音合成。
- **StableDiffusion**：角色立绘文生图。
- **LivePortrait**：立绘口型动态化。
- **视频合成技术**：包括但不限于视频剪辑、特效制作等。
- **视频高清化技术**: 扩增视频的分辨率。

## 安装与使用

### 环境要求

- Python 3.8+
- 其他依赖项请参考 `requirements.txt`

### 安装步骤
1. 安装视频处理包
   ```bash
   sudo apt update && sudo apt install ffmpeg imagemagick
   ```

<!--
sudo vim /etc/ImageMagick-6/policy.xml
##### change row to
<policy domain="path" rights="read|write" pattern="@*"/>
-->

2. 克隆仓库：
   ```bash
   git clone https://github.com/svjack/Genshin-Impact-Fan-Video.git
   ```

3. 进入项目目录：
   ```bash
   cd Genshin-Impact-Fan-Video
   ```

4. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 使用方法（整个处理流程在windows/linux系统下完成测试，生成视频前要确定moviepy、ffmpeg、magick等已成功安装配置）
### 以相亲场景二创短视频的制作为例

1. **生成相亲对话文案**：
- 在 `.cache/DeepSeek-key.txt` 中保存对应的 API Key。对应的 API Key 获取详见 [DeepSeek API](https://www.deepseek.com/)。
- 运行以下命令生成温迪的相亲对话文案：
```bash
python genshin_impact_meipo_dialogue_deepseek.py
```
- 在 `genshin_impact_meipo_dialogue_deepseek.py` 中，`genshin_impact_meipo_dialogue("温迪", False)` 这行代码是对温迪这个角色生成对应的相亲文案的执行代码。

### 温迪的相亲对话文案生成结果

1. *媒婆：你啥情况** - 媒婆询问男孩的背景。
2. *男孩：我是一个来路不明的吟游诗人\喜欢自由和热闹的气氛\讨厌奶酪和一切黏糊糊的玩意儿** - 男孩介绍自己的吟游诗人身份，喜欢自由和热闹，讨厌奶酪和黏糊糊的东西。
3. *媒婆：他是浪漫的灵魂\生活充满激情\对生活品质有独特追求** - 媒婆将男孩的吟游诗人身份解读为‘浪漫的灵魂’，喜欢自由和热闹的气氛被描述为‘生活充满激情’，讨厌奶酪和黏糊糊的东西则可以幽默地解释为‘对生活品质有独特追求’。
4. *女孩：这听起来挺特别的** - 女孩对男孩的背景表示好奇。
5. *媒婆：那你平时都做些什么** - 媒婆询问男孩的日常活动。
6. *男孩：我四处游历\唱唱歌\讲讲故事\有时候也会评论一下其他神明** - 男孩描述自己的日常活动，包括唱歌、讲故事和评论其他神明。
7. *媒婆：他是神明的批评家\对火之神和岩之神都有独到的见解** - 媒婆戏谑地称男孩为‘神明的批评家’，提到他对火之神和岩之神的评论。
8. *女孩：你具体是怎么评论的** - 女孩对男孩的评论感兴趣。
9. *男孩：火之神是个横行霸道的战斗狂\岩之神是个不懂人心的死脑筋** - 男孩表达对火之神和岩之神的偏见。
10. *媒婆：他的评论总是那么一针见血\让人忍俊不禁** - 媒婆称赞男孩的评论一针见血，幽默风趣。
11. *媒婆：那你现在住哪儿** - 媒婆询问男孩的居住情况。
12. *男孩：我居无定所\走到哪儿唱到哪儿** - 男孩描述自己居无定所的生活方式。
13. *媒婆：他是个真正的流浪诗人\生活自由自在\不受拘束** - 媒婆美化男孩的流浪生活，称其自由自在，不受拘束。
14. *女孩：那他有什么特别的爱好吗** - 女孩询问男孩的特别爱好。
15. *男孩：我喜欢收集各种奇怪的乐器\还有研究不同地方的民俗文化** - 男孩描述自己的特别爱好，包括收集乐器和研究民俗文化。
16. *媒婆：他的爱好丰富多彩\生活充满了趣味和探索** - 媒婆称赞男孩的爱好丰富多彩，生活充满趣味和探索。
17. *媒婆：那你有什么要求吗** - 媒婆询问男孩对女孩的要求。
18. *男孩：我希望她能理解我的生活方式\和我一起享受自由和热闹的气氛** - 男孩表达对女孩的要求，希望她能理解并享受他的生活方式。
19. *媒婆：他的浪漫灵魂是无价的\他的诗歌和故事就是最好的‘彩礼’** - 媒婆在讨论彩礼时开玩笑说，男孩的‘浪漫灵魂’是无价的，他的诗歌和故事就是最好的‘彩礼’。
20. *女孩：听起来很有趣** - 女孩对男孩的背景和要求表示感兴趣。
21. *媒婆：他是个不可多得的相亲对象\你可要好好把握机会** - 媒婆强调男孩的独特性格和背景，使他成为一个‘不可多得’的相亲对象。
22. *女孩：那彩礼方面呢** - 女孩开始讨论彩礼问题。
23. *媒婆：他的‘浪漫灵魂’是无价的\他的诗歌和故事就是最好的‘彩礼’** - 媒婆再次强调男孩的‘浪漫灵魂’是无价的，他的诗歌和故事就是最好的‘彩礼’。
24. *女孩：那我考虑一下** - 女孩表示会考虑。
25. *媒婆：考虑啥呀\这么好的小伙子\错过了可就没有了** - 媒婆催促女孩尽快决定，强调男孩的条件很好，错过了就没有了。
26. *女孩：那好吧\我同意了** - 女孩最终同意。
27. *媒婆：好嘞\那就这么定了** - 媒婆确认决定并推动事情进展。

<br/>

2. **生成静态短视频**：（76个角色）
- 运行以下命令生成静态短视频：
```bash
python genshin_impact_meipo_create_static_video.py
```

2. **生成动态短视频**：（这里以香菱、芭芭拉作为例子，且没有进行上采样，完整76个角色的高清化视频见下面示例视频中动态短视频的B站说明）
- 运行以下命令生成动态短视频：
```bash
python genshin_impact_meipo_create_dynamic_video.py
```

## 筛选后的文案与示例视频
### 媒婆文案
从多次生成结果中筛选后的文案上传至huggingface:
- [svjack/dialogue_feat_merge_save_unique](https://huggingface.co/datasets/svjack/dialogue_feat_merge_save_unique/tree/main) 中的以 _dia.txt 结尾的文件

### 静态短视频（打开视频声音按钮）

#### 温迪相亲视频

https://github.com/user-attachments/assets/63dec04c-2a5e-4bdf-8356-2bb3bcb3b707


生成的静态短视频结果已经上传至 Bilibili：
- [【温迪相亲记】](https://www.bilibili.com/video/BV1DvpieNENg/) 
- [76个角色连续的2小时合集版本](https://www.bilibili.com/video/BV1xCpiefEEq/)

<!--
其它75个人物的静态视频生成结果可于[斯温温jack](https://space.bilibili.com/3493273012275778)中的视频合集寻找
-->
其它75个人物的静态视频生成结果可于[原神媒婆的自我修养](https://space.bilibili.com/3493273012275778/channel/seriesdetail?sid=4360340&ctype=0)中的视频合集寻找

### 动态短视频（打开视频声音按钮）

#### 香菱相亲视频

https://github.com/user-attachments/assets/1213193d-34ad-4df6-a729-1216b70f834f


#### 芭芭拉相亲视频

https://github.com/user-attachments/assets/ef3e9d31-5cf0-4bf8-96d8-0464f82b952c

61个人物的动态视频生成结果可于[原神媒婆相亲二创](https://space.bilibili.com/3546775765911980/channel/seriesdetail?sid=4418073&ctype=0)中的视频合集寻找（新版本） <br/>
76个人物的动态视频生成结果可于[原神相亲记](https://space.bilibili.com/3493273012275778/channel/seriesdetail?sid=4416941&ctype=0)中的视频合集寻找（旧版本） <br/>
61个人物的动态视频生成结果也可于[svjack/Genshin-Impact-Meipo-Video](https://huggingface.co/datasets/svjack/Genshin-Impact-Meipo-Video)中查看（新版本） <br/>
<b>这些动态视频在上传Bilibili之前使用视频高清化技术进行了分辨率扩展</b> <br/>

## 上传到HuggingFace上的静态视频和动态视频数据集
| 数据集名称               | 链接                                                                 |
|--------------------------|--------------------------------------------------------------------|
| 原神媒婆静态短视频         | [https://huggingface.co/datasets/svjack/Genshin-Impact-Meipo-Static-Video](https://huggingface.co/datasets/svjack/Genshin-Impact-Meipo-Static-Video) |
| 原神媒婆动态短视频         | [https://huggingface.co/datasets/svjack/Genshin-Impact-Meipo-Video](https://huggingface.co/datasets/svjack/Genshin-Impact-Meipo-Video) |

## 创作灵感来源
### 相亲短视频
#### 灵感来源
本部分从《原神》二创视频《媒婆的语言魅力》（[点击查看](https://www.bilibili.com/video/BV12z421h71J/)）中汲取灵感。该系列是知名B站up主 Happy_Twins 的原神二创视频，视频的第一个合集为以原神须弥教令院学者珐露珊参与风系角色（包括：流浪者、万叶、温迪、魈、鹿野苑平藏）的若干相亲活动的故事。从现实生活中汲取灵感，利用珐露珊巧言善辩、聪明机警的灵活人际手腕，将若干风系角色或主角在相亲过程中的“问题”包装成“卖点”，进行成功推销。让我及很多人忍俊不禁、印象深刻的同时对于若干风系角色的生活遭遇（多为类似角色自身经历的现实演绎版本）感同身受乃至于怜悯同情，激发了我的创作灵感。

#### 创作基本思路

1. **文案生成**：
   - 从创作灵感的视频文案出发，将视频字幕结构化成对应的相亲文案。
   - 将三个相亲参与者抽象为男孩、媒婆、女孩三个人物身份。
   - 利用大语言模型作用于原神RAG知识系统（[Genshin-Impact-RAG](https://github.com/svjack/Genshin-Impact-RAG)），将原神人物对应于相亲男孩的身份，抽取其作为相亲对象的可能缺点，让大语言模型给出媒婆身份的解决方案，再根据解决方案生成对应的相亲对话结构化数据，完成AI文案制作。

2. **语音合成与静态视频制作**：
   - 对生成的文案进行拣选之后使用原神VITS模型给出对应的AI配音。
   - 结合角色的性格特征和游戏中的出现地点选取相对合适的对话背景图片和对话背景音乐。
   - 通过这些多媒体资源使用moviepy编程生成对应的短视频。

2. **动态视频制作**：
   - 根据角色立绘和一些画风设定使用StableDiffusion生成一些面部可动立绘图。
   - 对这些生成的立绘图片使用liveportrait技术生成对应的可变口型视频。
   - 结合角色的性格特征和游戏中的出现地点选取相对合适的原神风景视频和对话背景音乐。
   - 通过这些多媒体资源使用moviepy编程生成对应的短视频。

#### 本部分与 Happy_Twins 作品的区别

- **创作出发点**：本工程重点在文案生成和简单的结果短视频化，且创作出发点与Happy_Twins不同。Happy_Twins的角色经历多为生活中经历，一般是无法由游戏背景故事推测的，或者以娱乐为导向，其根源与角色设定关系点不太多，但结合其真人配音和MMD动作模组非常具有娱乐效果。
- **AI创作**：本工程从使用对原神知识特化的LLM RAG为起点，结合VITS技术，使用官方立绘和自己采集的照片生成静态短视频，使用文生图模型和liveportrait技术生成动态短视频，展示了AI创作的数量与效率优势。
- **私货**：工程中除了编程流程的工程化外，也夹带有若干私货，主要体现在生成的76个短视频中参与对象的关系。`genshin_impact_meipo_dialogue_make_video.py` 中通过 `relation_mapping` 定义了参与每一个视频演出的3个角色，其中“男孩”角色为76个游戏角色各一个，“女孩”角色则多为原神二创社区的CP磕糖（仅仅为了展示、不引战）或经常在剧情中交流的人物、在没有CP设定的情况下为本人自由发挥或自己认为应该组成一对儿的角色（私货）。背景图片都来源于本人在游戏中的拍摄取景，背景音乐的选取对于一些角色有采用相同旋律的不同音色来绑定二者的关系（私货），而且会将自己比较喜欢的旋律赋予比较喜欢的角色（私货）。

#### 常见CP包含

工程中包含的常见CP包含：
- 行重
- 钟魈
- 钟凝
- 五八
- 钟温
- 班雷
- 安柯
- 迪凯
- 绮娜
- 知妙
- 赛提
- 枫鹿
- 八雷
- 荒九
- 米诺
- 申甘
- 枫流
- 莱那
- 可雷
- 散妮
- 琴芭
- 七瑶
- 公钟

#### 角色私货

角色私货主要集中在：
- 芙宁娜
- 菲米尼
- 纳西妲
- 艾尔海森
- 刻晴
- 重云

## 贡献

欢迎大家贡献代码、提出建议或报告问题。请参考 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

## 许可证

本项目采用 MIT 许可证。详细信息请参考 [LICENSE](LICENSE) 文件。

## 联系我们

如有任何问题或建议，请通过 [ehangzhou@outlook.com](ehangzhou@outlook.com) 联系我们。

---

**Genshin-Impact-Fan-Video** 是一个充满创意和幽默的项目，旨在为《原神》玩家带来更多欢乐。感谢您的关注和支持！
