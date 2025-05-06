import os
from pathlib import Path
import json
from tqdm import tqdm
import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

from pycocotools.coco import COCO
from datasets import load_dataset
from loguru import logger
from datetime import datetime


def extract_coco_matches(doc, matcher):
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    filtered_spans = filter_spans(spans)

    results = {"labels": [], 
               "label_positions": []}
    
    for span in filtered_spans:
        phrase = span.text
        start = span.start_char
        end = span.end_char
        results["labels"].append(phrase)
        results["label_positions"].append([start, end])

    return results["labels"], results["label_positions"]


def main(args):
    logger.info(f"Configuration: split={args.split}, output_dir={args.output_dir}")

    # Load Visual Spatial Reasoning (VSR) dataset from Hugging Face
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    logger.info(f"Loading VSR dataset from Hugging Face (split: {args.split})")
    vsp_dataset = load_dataset(f"cambridgeltl/vsr_{args.split}", data_files=data_files)

    # Load COCO annotations
    coco_train = COCO(args.coco_instances_train_file)
    logger.info(f"Loaded COCO train annotations from {args.coco_instances_train_file}")
    logger.info(f"Number of COCO train annotations: {len(coco_train.anns)}")

    coco_val = COCO(args.coco_instances_val_file)
    logger.info(f"Loaded COCO val annotations from {args.coco_instances_val_file}")
    logger.info(f"Number of COCO val annotations: {len(coco_val.anns)}")

    cats = coco_train.loadCats(coco_train.getCatIds())
    category_to_id = {cat['name']: cat['id'] for cat in cats}
    all_categories = list(category_to_id.keys())
    logger.info(f"Number of COCO categories: {len(all_categories)}")

    # Setup spaCy matcher
    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(cat) for cat in all_categories]
    matcher.add("COCO", patterns)

    # Main loop
    get_image_id = lambda x: int(x.split(".")[0].lstrip('0'))

    total_skipped = 0
    total_samples = 0
    for split in data_files.keys():
        logger.info(f"Processing {split} split")

        file_name = f"{split}.jsonl"
        output_file = os.path.join(args.output_dir, file_name)

        data = vsp_dataset[split]

        skipped = 0
        for sample in tqdm(data, disable=True):

            # Ensure image_id is valid
            image_id: int = get_image_id(sample["image"])

            # Ensure image_id is in coco_train or coco_val
            if image_id not in coco_train.imgs:
                if image_id not in coco_val.imgs:
                    logger.warning(f"Skipping sample {image_id}: Invalid image ID")
                    skipped += 1
                    continue
                else:
                    coco = coco_val
            else:
                coco = coco_train

            img_metadata = coco.loadImgs(image_id)[0]

            # Extract labels from caption                
            caption = sample["caption"]
            doc = nlp(caption)
            labels, label_positions = extract_coco_matches(doc, matcher)

            if len(labels) != 2:
                logger.warning(f"Skipping sample {image_id}: Found {len(labels)} labels in caption '{caption}' (expected 2)")
                skipped += 1
                continue

            # Retrieve corresponding bboxes
            ann_ids = coco.getAnnIds(imgIds=img_metadata['id'])
            anns = coco.loadAnns(ann_ids)

            bbox_1 = None
            bbox_2 = None
            for ann in anns:
                if ann['category_id'] == category_to_id[labels[0]]:
                    if bbox_1 is not None:
                        logger.warning(f"Skipping sample {image_id}: Multiple bboxes found for {labels[0]}")
                        skipped += 1
                        continue
                    bbox_1 = ann['bbox']
                elif ann['category_id'] == category_to_id[labels[1]]:
                    if bbox_2 is not None:
                        logger.warning(f"Skipping sample {image_id}: Multiple bboxes found for {labels[1]}")
                        skipped += 1
                        continue
                    bbox_2 = ann['bbox']

            # Check if bbox_1 and bbox_2 are valid
            if bbox_1 is None or bbox_2 is None:
                logger.warning(f"Skipping sample {image_id}: Missing bboxes for labels {labels}")
                skipped += 1
                continue

            ref_exp = {
                "labels": labels,
                "label_positions": label_positions,
                "bboxes": [bbox_1, bbox_2]
            }

            data_item = {
                "image_file": sample["image"],
                "image_link": sample["image_link"],
                "width": img_metadata["width"],
                "height": img_metadata["height"],
                "caption": sample["caption"],
                "label": sample["label"],
                "relation": sample["relation"],
                "ref_exp": ref_exp,
            }
            
            # Write to file
            with open(output_file, "a") as f:
                f.write(json.dumps(data_item) + "\n")
                total_samples += 1
        
        total_skipped += skipped

        logger.info(f"Completed {split} split ({total_skipped} skipped samples)")
    
    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Total skipped samples: {total_skipped}")


if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_instances_train_file", type=str, required=True, help="Path to COCO train annotations (.json file)")
    parser.add_argument("--coco_instances_val_file", type=str, required=True, help="Path to COCO val annotations (.json file)")
    parser.add_argument("--split", type=str, required=True, choices=["zeroshot", "random"], help="Split to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    output_dir = f"./output/{args.split}"
    if os.path.exists(output_dir):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Directory {output_dir} already exists")
        
    os.makedirs(output_dir)

    args.output_dir = output_dir
    
    # Setup logging
    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log"
    logger.add(log_file, level="INFO")

    main(args)
