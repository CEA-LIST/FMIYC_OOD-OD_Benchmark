from collections import defaultdict
import json

SOURCE_FILE_NAME = "../CVDatasets/OpenImages/ood_classes_rm_overlap/COCO-Format/farther_oi_wrt_bdd.json"
TARGET_FILE_NAME = "../CVDatasets/OpenImages/ood_classes_rm_overlap/COCO-Format/farther_oi_wrt_bdd_int_id.json"


def main(source_file_name, target_file_name) -> None:
    """
    This functions takes as input a COCO annotations file that has alphanumeric ids and saves one where the
    ids are integers

    :return:
    """
    print(f"Editing {source_file_name}")
    # Load json and instantiate parser
    openim_annotations_file = source_file_name
    # Instantiate coco parser
    coco = COCOParser(openim_annotations_file)

    # Get images ids
    img_ids = coco.get_imgIds()
    ann_ids = coco.get_annIds(img_ids)
    anns = coco.load_anns(ann_ids)
    imgs_info = coco.get_img_info(img_ids)

    # Build new dictionary
    int_id_coco_dict = {
        'info': coco.info_dict['info'],
        'licenses': coco.licenses_dict['licenses'],
        'images': imgs_info,
        'annotations': anns,
        'categories': coco.categories_original['categories']
    }
    # Save dictionary as json
    with open(target_file_name, "w") as outfile:
        json.dump(int_id_coco_dict, outfile)

    print(f"Saved {len(imgs_info)} images in {target_file_name}")


class COCOParser:
    def __init__(self, anns_file):
        with open(anns_file, 'r') as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        # Dict of id: category pairs
        self.cat_dict = {}
        # Dict of original categories (copy of original entry)
        self.categories_original = {'categories': coco['categories']}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {'licenses': coco['licenses']}
        self.info_dict = {'info': coco['info']}
        im_id_int_dict = {}
        for idx, img in enumerate(coco['images']):
            im_id_int_dict[img['id']] = idx + 1
            img['id'] = idx + 1
            # Remove leading zeros from filenames
            # try:
            #     img['file_name'] = str(int(img['file_name'].split('.')[0])) + '.jpg'
            # except ValueError:
            #     pass
            self.im_dict[idx + 1] = img
        for ann in coco['annotations']:
            im_id = ann['image_id']
            ann['image_id'] = im_id_int_dict[im_id]
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']] = ann

        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        # Licenses not actually needed per image
        # for license in coco['licenses']:
        #     self.licenses_dict[license['id']] = license

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        im_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def load_cats(self, class_ids):
        class_ids = class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def get_imgLicenses(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]

    def get_img_info(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [self.im_dict[im_id] for im_id in im_ids]

if __name__ == '__main__':
    main(SOURCE_FILE_NAME, TARGET_FILE_NAME)
