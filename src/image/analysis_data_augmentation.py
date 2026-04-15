import cv2
import gc
import time
import numpy as np
import random

from core.service_base import Visualization
from core.data_base import Object
from image.dataset import ImageDataset
from config import CLASS_NAMES, PATH_FOLDER_RAW
import os
from image.preprocessing_data_augmentation import (
    horizontal_flip, rotate_image, random_crop, add_gaussian_noise, adjust_brightness_contrast,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# TN1: 8 phep bien doi don le
SINGLE_TRANSFORMS = [
    ("Horizontal Flip",      lambda img: horizontal_flip(img)),
    ("Rotate",            lambda img: rotate_image(img, angle=15)),
    ("Random Crop",      lambda img: random_crop(img, crop_factor=0.8)),
    ("Gaussian Noise",  lambda img: add_gaussian_noise(img, mean=0, std=20)),
    ("Brightness",     lambda img: adjust_brightness_contrast(img, alpha=1.3, beta=20)),
]

# TN3: 2 bien the cho moi phep
DUAL_VARIANTS = {
    "Rotate": [
        ("Lead_Rotate_+20", lambda img: horizontal_flip(add_gaussian_noise(random_crop(adjust_brightness_contrast(rotate_image(img, angle=20), alpha=1.1, beta=10), crop_factor=0.85), mean=0, std=15))),
        ("Lead_Rotate_-20", lambda img: horizontal_flip(add_gaussian_noise(random_crop(adjust_brightness_contrast(rotate_image(img, angle=-20), alpha=1.1, beta=10), crop_factor=0.85), mean=0, std=15)))
    ],
    
    "Random Crop": [
        ("Lead_Crop_0.90", lambda img: horizontal_flip(add_gaussian_noise(rotate_image(adjust_brightness_contrast(random_crop(img, crop_factor=0.90), alpha=1.1, beta=10), angle=10), mean=0, std=15))),
        ("Lead_Crop_0.60", lambda img: horizontal_flip(add_gaussian_noise(rotate_image(adjust_brightness_contrast(random_crop(img, crop_factor=0.60), alpha=1.1, beta=10), angle=10), mean=0, std=15)))
    ],
    
    "Gaussian Noise": [
        ("Lead_Noise_s10", lambda img: horizontal_flip(random_crop(rotate_image(adjust_brightness_contrast(add_gaussian_noise(img, mean=0, std=10), alpha=1.1, beta=10), angle=10), crop_factor=0.85))),
        ("Lead_Noise_s30", lambda img: horizontal_flip(random_crop(rotate_image(adjust_brightness_contrast(add_gaussian_noise(img, mean=0, std=30), alpha=1.1, beta=10), angle=10), crop_factor=0.85)))
    ],
    
    "Brightness/Contrast": [
        ("Lead_Bright_a1.3", lambda img: horizontal_flip(add_gaussian_noise(random_crop(rotate_image(adjust_brightness_contrast(img, alpha=1.3, beta=20), angle=10), crop_factor=0.85), mean=0, std=15))),
        ("Lead_Dark_a0.7",   lambda img: horizontal_flip(add_gaussian_noise(random_crop(rotate_image(adjust_brightness_contrast(img, alpha=0.7, beta=-20), angle=10), crop_factor=0.85), mean=0, std=15)))
    ]
} 

TRANSFORM_KEYS = list(DUAL_VARIANTS.keys())

# TN2: to hop 3 phep bien doi tuan tu
COMBO_3_PIPELINES = [
    # 1. HFlip + Rotate + Crop
    ("HFlip-Rotate-Crop",
     [lambda img: horizontal_flip(img),
      lambda img: rotate_image(img, angle=15),
      lambda img: random_crop(img, crop_factor=0.85)]),

    # 2. HFlip + Rotate + Noise
    ("HFlip-Rotate-Noise",
     [lambda img: horizontal_flip(img),
      lambda img: rotate_image(img, angle=-15),
      lambda img: add_gaussian_noise(img, mean=0, std=20)]),

    # 3. HFlip + Rotate + Brightness
    ("HFlip-Rotate-Bright",
     [lambda img: horizontal_flip(img),
      lambda img: rotate_image(img, angle=20),
      lambda img: adjust_brightness_contrast(img, alpha=1.2, beta=15)]),

    # 4. HFlip + Crop + Noise
    ("HFlip-Crop-Noise",
     [lambda img: horizontal_flip(img),
      lambda img: random_crop(img, crop_factor=0.80),
      lambda img: add_gaussian_noise(img, mean=0, std=15)]),

    # 5. HFlip + Crop + Brightness
    ("HFlip-Crop-Dark",
     [lambda img: horizontal_flip(img),
      lambda img: random_crop(img, crop_factor=0.90),
      lambda img: adjust_brightness_contrast(img, alpha=0.8, beta=-15)]),

    # 6. HFlip + Noise + Brightness
    ("HFlip-Noise-Bright",
     [lambda img: horizontal_flip(img),
      lambda img: add_gaussian_noise(img, mean=0, std=20),
      lambda img: adjust_brightness_contrast(img, alpha=1.3, beta=10)]),

    # 7. Rotate + Crop + Noise
    ("Rotate-Crop-Noise",
     [lambda img: rotate_image(img, angle=15),
      lambda img: random_crop(img, crop_factor=0.85),
      lambda img: add_gaussian_noise(img, mean=0, std=25)]),

    # 8. Rotate + Crop + Brightness
    ("Rotate-Crop-Bright",
     [lambda img: rotate_image(img, angle=-20),
      lambda img: random_crop(img, crop_factor=0.80),
      lambda img: adjust_brightness_contrast(img, alpha=1.1, beta=10)]),

    # 9. Rotate + Noise + Brightness
    ("Rotate-Noise-Dark",
     [lambda img: rotate_image(img, angle=25),
      lambda img: add_gaussian_noise(img, mean=0, std=15),
      lambda img: adjust_brightness_contrast(img, alpha=0.9, beta=-10)]),

    # 10. Crop + Noise + Brightness
    ("Crop-Noise-Bright",
     [lambda img: random_crop(img, crop_factor=0.85),
      lambda img: add_gaussian_noise(img, mean=0, std=20),
      lambda img: adjust_brightness_contrast(img, alpha=1.2, beta=20)])
]

def _apply_pipeline(image, fn_list):
    result = image.copy()
    for fn in fn_list:
        result = fn(result)
    return result

class AugmentationExperiment(Visualization):
    def __init__(self, target_size=(64, 64), num_eval_runs=2):
        self.step_name     = "Data Augmentation Experiment"
        self.dataset_name  = "Unknown"
        self.status        = "Pending"
        self.target_size   = target_size
        self.num_eval_runs = num_eval_runs

        self.all_images  = None
        self.all_paths   = None
        self.all_labels  = None
        self.tsne_labels = None

        self.results_exp1 = None
        self.results_exp2 = None
        self.results_exp3 = None

        self.eval_results = None
        self.train_idx    = None
        self.test_idx     = None

    def run(self, obj: Object):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        else:
            self.status = "Failed (Not an ImageDataset)"
            self.log()

    def visitImageDataset(self, obj: ImageDataset):
        self.dataset_name = obj.folder_path or "ImageDataset"
        try:
            self._load_data(obj)
            self._split_data()
            self._evaluate_all()
            self.status = "Success"
        except Exception as e:
            self.status = f"Failed ({str(e)})"
            import traceback
            traceback.print_exc()
        finally:
            self.log()

    def log(self):
        print(f"\n{'='*55}")
        print(f"  Buoc xu ly   : {self.step_name}")
        print(f"  Tap du lieu  : {self.dataset_name}")
        print(f"  Resize       : {self.target_size}")
        print(f"  Trang thai   : {self.status}")
        if self.status == "Success":
            print(f"  Tong anh     : {len(self.all_images)}")
            print(f"  Train/Test   : {len(self.train_idx)}/{len(self.test_idx)}")
            print(f"  TN1          : {len(self.results_exp1)}")
            print(f"  TN2          : {len(self.results_exp2)}")
            print(f"  TN3          : {len(self.results_exp3)} pipeline uu tien")
            if self.eval_results:
                baseline = next(r for r in self.eval_results if r['group'] == 'Baseline')
                best     = max(self.eval_results, key=lambda r: r['f1'])
                print(f"  Baseline F1  : {baseline['f1']:.4f}")
                print(f"  Best F1      : {best['f1']:.4f}  ({best['name']})")
        print(f"{'='*55}")
    
    def _save_augmented_images(self, tn_group, method_name, aug_imgs, aug_labels, aug_paths):
        safe_method_name = method_name.replace("/", "-").replace(" ", "_").replace(":", "")
        base_dir = os.path.join(PATH_FOLDER_RAW, "preprocessing", tn_group, safe_method_name)
        for i, img in enumerate(aug_imgs):
            class_name = str(CLASS_NAMES[aug_labels[i]]).replace(" ", "_")
            basename = os.path.basename(aug_paths[i])
            safe_basename = basename.replace(" ", "_")
            out_dir = os.path.join(base_dir, class_name)
            os.makedirs(out_dir, exist_ok=True)
            
            
            rand_suffix = str(random.randint(1000, 9999))
            final_name = f"{rand_suffix}_{safe_basename}"
            out_path = os.path.join(out_dir, final_name)
            try:
                is_success, im_buf_arr = cv2.imencode(".jpg", img)
                if is_success:
                    im_buf_arr.tofile(out_path)
            except Exception as e:
                pass

    def _load_data(self, obj: ImageDataset):
        images_temp = []
        labels_temp = []
        paths_temp  = []
        total     = len(obj.image_paths)
        processed = 0

        for batch_imgs, batch_indices in obj.load():
            for img, idx in zip(batch_imgs, batch_indices):
                img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
                images_temp.append(img_resized)
                labels_temp.append(obj._labels[idx])
                paths_temp.append(obj.image_paths[idx])
            processed += len(batch_imgs)
            print(f"  -> Da load: {processed} / {total} anh...", end='\r')
            del batch_imgs
            gc.collect()

        print(f"\n[INFO] Load hoan tat: {processed} anh -> resize ve {self.target_size}.")
        self.all_images  = images_temp
        self.all_labels  = labels_temp
        self.all_paths   = paths_temp
        self.tsne_labels = [lbl - 1 for lbl in labels_temp]

    def _split_data(self):
        indices = np.arange(len(self.all_images))
        self.train_idx, self.test_idx = train_test_split(
            indices, test_size=0.2, random_state=42,
            stratify=self.tsne_labels
        )
        print(f"[INFO] Train: {len(self.train_idx)}, Test: {len(self.test_idx)}")

    def evaluation(self, X_train, y_train, X_test, y_test):
        acc_list, prec_list, rec_list, f1_list, time_list = [], [], [], [], []
        for i in range(self.num_eval_runs):
            start = time.time()
            lr = LogisticRegression(
                max_iter=200,
                random_state=42 + i,
                solver='lbfgs',
                multi_class='auto',
                n_jobs=-1
            )
            lr.fit(X_train, y_train)
            elapsed = time.time() - start
            y_pred  = lr.predict(X_test)

            acc_list.append(accuracy_score(y_test, y_pred))
            prec_list.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            rec_list.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            f1_list.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            time_list.append(elapsed)

        return {
            'accuracy':     round(float(np.mean(acc_list)),  4),
            'precision':    round(float(np.mean(prec_list)), 4),
            'recall':       round(float(np.mean(rec_list)),  4),
            'f1':           round(float(np.mean(f1_list)),   4),
            'train_time':   round(float(np.mean(time_list)), 3),
            'accuracy_std': round(float(np.std(acc_list)),   4),
            'f1_std':       round(float(np.std(f1_list)),    4),
        }

    def _evaluate_pipeline_balance(self, pipeline_fn):
        import random
        from collections import Counter
        
        X_train = [self.all_images[i] for i in self.train_idx]
        y_train = [self.tsne_labels[i] for i in self.train_idx]
        train_paths = [self.all_paths[i] for i in self.train_idx]
        
        counts = Counter(y_train)
        max_count = max(counts.values()) if counts else 0
        
        # 1. Cân bằng các class
        X_bal_add = []
        y_bal_add = []
        path_bal_add = []
        
        random.seed(42)
        for c in counts:
            needed = max_count - counts[c]
            if needed > 0:
                indices_c = [i for i, lbl in enumerate(y_train) if lbl == c]
                picked = random.choices(indices_c, k=needed)
                for orig_idx in picked:
                    a_img = pipeline_fn(X_train[orig_idx].copy())
                    X_bal_add.append(a_img)
                    y_bal_add.append(c)
                    path_bal_add.append(train_paths[orig_idx])
                    
        # Tập cân bằng(Dataset gốc + ảnh được augment)
        X_bal_full = X_train + X_bal_add
        y_bal_full = y_train + y_bal_add
        path_bal_full = train_paths + path_bal_add
        
        synth_imgs = X_bal_add
        synth_labels = y_bal_add
        synth_paths = path_bal_add
        
        # Chuẩn bị ma trận đánh giá cho LR
        X_train_flat = [img.flatten().astype(np.float32) / 255.0 for img in X_bal_full]
        y_train_final = np.array(y_bal_full)
        
        # T-SNE sẽ dùng Train_final + Test_final
        X_test = [self.all_images[i] for i in self.test_idx]
        X_test_flat = [img.flatten().astype(np.float32) / 255.0 for img in X_test]
        y_test = [self.tsne_labels[i] for i in self.test_idx]
        
        tsne_imgs = X_bal_full + X_test
        tsne_labels = np.concatenate([y_train_final, y_test])
        
        metrics = self.evaluation(np.array(X_train_flat), y_train_final, np.array(X_test_flat), np.array(y_test))
        metrics['train_size'] = len(X_train_flat)
        return synth_imgs, synth_labels, synth_paths, tsne_imgs, tsne_labels, metrics

    def _evaluate_all(self):
        results = []
        
        X_test  = [self.all_images[i] for i in self.test_idx]
        X_test_flat = [img.flatten().astype(np.float32) / 255.0 for img in X_test]
        y_test = np.array([self.tsne_labels[i] for i in self.test_idx])
        
        X_train = [self.all_images[i] for i in self.train_idx]
        X_train_flat = [img.flatten().astype(np.float32) / 255.0 for img in X_train]
        y_train = np.array([self.tsne_labels[i] for i in self.train_idx])

        # Baseline
        metrics = self.evaluation(np.array(X_train_flat), y_train, np.array(X_test_flat), y_test)
        metrics.update({
            'name':             'Baseline (khong augment)',
            'group':            'Baseline',
            'n_transforms':     0,
            'n_pipeline_steps': 0,
            'train_size':       len(X_train),
        })
        results.append(metrics)
        print(f"  [Baseline] F1={metrics['f1']:.4f}")

        # TN1
        print("[PROCESS] TN1: Ap dung tung phep bien doi")
        self.results_exp1 = []
        for name, fn in SINGLE_TRANSFORMS:
            sym_imgs, sym_lbls, sym_paths, tsne_imgs, tsne_lbls, m = self._evaluate_pipeline_balance(fn)
            self._save_augmented_images('tn_1', name, sym_imgs, sym_lbls, sym_paths)
            self.results_exp1.append((name, tsne_imgs, tsne_lbls))
            
            m.update({'name': f'TN1: {name}', 'group': 'TN1', 'n_transforms': 1, 'n_pipeline_steps': 1})
            results.append(m)
            print(f"  [TN1] {name:28s} F1={m['f1']:.4f}")

        # TN2
        print("[PROCESS] TN2: To hop 3 phep")
        self.results_exp2 = []
        for name, pipeline in COMBO_3_PIPELINES:
            def p_fn(img, p_list=pipeline):
                return _apply_pipeline(img, p_list)
                
            sym_imgs, sym_lbls, sym_paths, tsne_imgs, tsne_lbls, m = self._evaluate_pipeline_balance(p_fn)
            self._save_augmented_images('tn_2', name, sym_imgs, sym_lbls, sym_paths)
            self.results_exp2.append((name, tsne_imgs, tsne_lbls))
            
            m.update({'name': f'TN2: {name}', 'group': 'TN2', 'n_transforms': 3, 'n_pipeline_steps': 3})
            results.append(m)
            print(f"  [TN2] {name:28s} F1={m['f1']:.4f}")

        # TN3
        print("[PROCESS] TN3: Pipeline uu tien")
        n_steps = 2 + (len(TRANSFORM_KEYS) - 1) * 2
        self.results_exp3 = []
        for lead_key, variants in DUAL_VARIANTS.items():
            for name, pipeline_fn in variants:
                sym_imgs, sym_lbls, sym_paths, tsne_imgs, tsne_lbls, m = self._evaluate_pipeline_balance(pipeline_fn)
                self._save_augmented_images('tn_3', name, sym_imgs, sym_lbls, sym_paths)
                self.results_exp3.append((name, tsne_imgs, tsne_lbls))
                
                m.update({'name': f'TN3: {name}', 'group': 'TN3', 'n_transforms': len(TRANSFORM_KEYS), 'n_pipeline_steps': n_steps})
                results.append(m)
                print(f"  [TN3] {name:28s} F1={m['f1']:.4f}")

        self.eval_results = results

    def resize_for_tsne(self, images):
        results = []
        for img in images:
            if img.shape[:2] != (self.target_size[1], self.target_size[0]):
                results.append(cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR))
            else:
                results.append(img)
        return results
