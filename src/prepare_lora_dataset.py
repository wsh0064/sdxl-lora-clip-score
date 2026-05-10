import os
import json
import traceback
from pathlib import Path

from logger_utils import setup_logger

logger = setup_logger("prepare_dataset")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CAPTION_JSON = PROJECT_ROOT / "data" / "annotations_trainval2017" / "annotations" / "captions_train2017.json"
TRAIN_IMAGES_DIR = PROJECT_ROOT / "data" / "train2017"
OUTPUT_DIR = PROJECT_ROOT / "data" / "train_lora_dataset"


def main():
    logger.info("=" * 60)
    logger.info("开始准备 LoRA 训练数据集")
    logger.info("=" * 60)
    logger.info(f"标注文件: {CAPTION_JSON}")
    logger.info(f"图片目录: {TRAIN_IMAGES_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info("=" * 60)

    try:
        logger.info(f"加载 COCO 标注文件...")
        with open(CAPTION_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        image_id_to_first_caption = {}

        for ann in data["annotations"]:
            iid = ann["image_id"]
            if iid not in image_id_to_first_caption:
                image_id_to_first_caption[iid] = ann["caption"]

        logger.info(f"找到 {len(image_id_to_file)} 张图片，{len(image_id_to_first_caption)} 张有标注")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        metadata_path = OUTPUT_DIR / "metadata.jsonl"

        created_count = 0
        skipped_count = 0

        logger.info(f"创建数据集链接和元数据...")
        with open(metadata_path, "w", encoding="utf-8") as out_f:
            for iid, file_name in image_id_to_file.items():
                if iid not in image_id_to_first_caption:
                    skipped_count += 1
                    continue

                caption = image_id_to_first_caption[iid]
                src_path = TRAIN_IMAGES_DIR / file_name

                if not src_path.exists():
                    skipped_count += 1
                    continue

                dst_path = OUTPUT_DIR / file_name
                if not dst_path.exists():
                    dst_path.symlink_to(src_path)

                line = json.dumps({"file_name": file_name, "text": caption}, ensure_ascii=False)
                out_f.write(line + "\n")
                created_count += 1

        logger.info("=" * 60)
        logger.info("数据集准备完成！")
        logger.info(f"输出目录: {OUTPUT_DIR}")
        logger.info(f"元数据: {metadata_path}")
        logger.info(f"成功创建: {created_count} 个样本")
        logger.info(f"跳过: {skipped_count} 个样本")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error("=" * 60)
        logger.error(f"文件未找到: {str(e)}")
        logger.error("请确保 COCO 数据集已正确放置")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)


if __name__ == "__main__":
    main()
