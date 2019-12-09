# 机器学习大作业
#### 算法使用

YOLOv3算法

#### 相关依赖

```
pip install -r requirements.txt
```

#### 测试文件

```
python test.py --img_set_path "data/valid.txt" --img_path "data/images/" --anno_path "data/labels/"
```

**img_set_path：** 存放图片名的文本文件所在路径。文件内容为每行一条文件名，文件名不包含路径与文件后缀名。

**img_path：** 图片所在路径。通过逐行读取img_set_path内容，将图片名与路径拼接，并加上图片后缀名('.jpg')，获得图片全路径。

**anno_path：** 图片注解所在路径。图片注解名与图片名相同，将图片名与路径拼接，并加上注解文件后缀名('.txt')，获得图片注解文件全路径。

