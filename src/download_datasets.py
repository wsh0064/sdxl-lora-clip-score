#!/usr/bin/env python3
"""
自动下载项目所需数据集和预训练模型脚本
支持断点续传、进度显示、文件完整性校验
"""
import os
import zipfile
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 下载资源配置
DOWNLOAD_CONFIG = [
    {
        "name": "COCO 2017 标注文件",
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "save_path": PROJECT_ROOT / "data" / "annotations_trainval2017.zip",
        "extract_path": PROJECT_ROOT / "data",
        "md5": "8551ee4bb5860311e79dace7e79cb91e",
        "required": True
    },
    {
        "name": "COCO 2017 验证集 (5GB)",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "save_path": PROJECT_ROOT / "data" / "val2017.zip",
        "extract_path": PROJECT_ROOT / "data",
        "md5": "442b8da769d5846986a4a73a8ab49c20",
        "required": True
    },
    {
        "name": "FID InceptionV3 预训练模型",
        "url": "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth",
        "save_path": PROJECT_ROOT / "models" / "pt_inception-2015-12-05-6726825d.pth",
        "extract_path": None,
        "md5": "3f6e3d86c77a7d9019e4d088f1e3d9c2",
        "required": True
    }
]

def calculate_md5(file_path, chunk_size=4096):
    """计算文件MD5值用于校验"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def download_file(url, save_path, desc="下载中"):
    """断点续传下载文件"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 获取已下载文件大小
    resume_byte_pos = 0
    if save_path.exists():
        resume_byte_pos = save_path.stat().st_size
    
    headers = {"Range": f"bytes={resume_byte_pos}-"} if resume_byte_pos > 0 else {}
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    
    # 获取总文件大小
    total_size = int(response.headers.get('content-length', 0))
    if resume_byte_pos > 0:
        total_size += resume_byte_pos
    
    # 打开文件，追加模式
    mode = 'ab' if resume_byte_pos > 0 else 'wb'
    with open(save_path, mode) as f, tqdm(
        desc=desc,
        total=total_size,
        initial=resume_byte_pos,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                size = f.write(chunk)
                pbar.update(size)
    
    return save_path

def extract_zip(zip_path, extract_path):
    """解压zip文件"""
    extract_path.mkdir(parents=True, exist_ok=True)
    print(f"正在解压: {zip_path.name} -> {extract_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 获取文件总数用于进度显示
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        
        for file in tqdm(file_list, desc="解压进度"):
            zip_ref.extract(file, extract_path)
    
    print(f"解压完成: {extract_path}")
    # 可选：删除压缩包节省空间
    # os.remove(zip_path)
    # print(f"已删除压缩包: {zip_path}")

def main():
    print("="*60)
    print("SDXL LoRA 项目数据集自动下载工具")
    print("="*60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print()
    
    for item in DOWNLOAD_CONFIG:
        name = item["name"]
        url = item["url"]
        save_path = Path(item["save_path"])
        extract_path = Path(item["extract_path"]) if item["extract_path"] else None
        expected_md5 = item["md5"]
        required = item["required"]
        
        print(f"\n📦 处理: {name}")
        
        # 检查文件是否已存在且完整
        if save_path.exists():
            print(f"🔍 校验现有文件: {save_path.name}")
            file_md5 = calculate_md5(save_path)
            if file_md5 == expected_md5:
                print("✅ 文件已存在且校验通过，跳过下载")
                if extract_path and not any(extract_path.iterdir()):
                    extract_zip(save_path, extract_path)
                continue
            else:
                print("⚠️ 文件校验不通过，重新下载")
        
        # 下载文件
        try:
            download_file(url, save_path, desc=f"下载 {name}")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            if required:
                print("错误：必需资源下载失败，退出")
                return
            else:
                print("警告：非必需资源下载失败，跳过")
                continue
        
        # 校验下载后的文件
        print("🔍 校验文件完整性...")
        file_md5 = calculate_md5(save_path)
        if file_md5 != expected_md5:
            print(f"❌ 文件校验失败，MD5不匹配: 预期 {expected_md5}，实际 {file_md5}")
            os.remove(save_path)
            if required:
                return
            continue
        
        print("✅ 文件校验通过")
        
        # 解压
        if extract_path:
            try:
                extract_zip(save_path, extract_path)
            except Exception as e:
                print(f"❌ 解压失败: {e}")
                return
    
    print("\n" + "="*60)
    print("✅ 所有资源下载完成！")
    print("="*60)
    print(f"数据集目录: {PROJECT_ROOT / 'data'}")
    print(f"模型目录: {PROJECT_ROOT / 'models'}")
    print("\n现在可以开始训练和评估了~")

if __name__ == "__main__":
    main()
