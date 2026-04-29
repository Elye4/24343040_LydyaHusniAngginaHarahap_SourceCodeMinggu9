import numpy as np
import cv2
import matplotlib.pyplot as plt

def praktikum_9_1():
    print("PRAKTIKUM 9.1: PERBANDINGAN TEKNIK THRESHOLDING")
    print("=" * 60)
    
    # =========================
    # Membuat Citra Uji
    # =========================
    def create_test_images():
        images = {}
        
        img_bimodal = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_bimodal, (30, 30), (150, 150), 50, -1)
        cv2.rectangle(img_bimodal, (100, 100), (220, 220), 200, -1)
        images['Bimodal Image'] = img_bimodal
        
        img_uneven = np.zeros((256, 256), dtype=np.uint8)
        for i in range(256):
            img_uneven[:, i] = i // 2
        cv2.rectangle(img_uneven, (50, 50), (100, 100), 255, -1)
        cv2.rectangle(img_uneven, (150, 150), (200, 200), 100, -1)
        images['Uneven Illumination'] = img_uneven
        
        img_noisy = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_noisy, (50, 50), (150, 150), 128, -1)
        noise = np.random.normal(0, 30, img_noisy.shape)
        img_noisy = np.clip(img_noisy + noise, 0, 255).astype(np.uint8)
        images['Noisy Image'] = img_noisy
        
        img_multi = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_multi, (30, 30), (90, 90), 80, -1)
        cv2.rectangle(img_multi, (100, 30), (160, 90), 120, -1)
        cv2.rectangle(img_multi, (170, 30), (230, 90), 180, -1)
        images['Multi-level Image'] = img_multi
        
        return images

    # =========================
    # Thresholding
    # =========================
    def apply_global_threshold(image, T=127):
        _, binary = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
        return binary
    
    def apply_otsu_threshold(image):
        _, binary = cv2.threshold(image, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def apply_adaptive_threshold(image):
        return cv2.adaptiveThreshold(image, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    
    def apply_iterative_threshold(image):
        T = np.mean(image)
        for _ in range(100):
            fg = image[image > T]
            bg = image[image <= T]
            if len(fg) == 0 or len(bg) == 0:
                break
            T_new = (np.mean(fg) + np.mean(bg)) / 2
            if abs(T - T_new) < 1:
                break
            T = T_new
        
        _, binary = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
        return binary, T

    # =========================
    # Proses
    # =========================
    test_images = create_test_images()
    results = {}

    for name, gray in test_images.items():
        global_b = apply_global_threshold(gray)
        otsu_b = apply_otsu_threshold(gray)
        adaptive_b = apply_adaptive_threshold(gray)
        iterative_b, T_iter = apply_iterative_threshold(gray)

        # FIX OTSU VALUE
        T_otsu, _ = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        results[name] = {
            'original': gray,
            'global': global_b,
            'otsu': otsu_b,
            'adaptive': adaptive_b,
            'iterative': iterative_b,
            'T_otsu': T_otsu,
            'T_iter': T_iter
        }

    # =========================
    # 1. VISUALISASI UTAMA (20 gambar)
    # =========================
    fig1, axes = plt.subplots(4, 5, figsize=(20, 16))

    for i, (name, res) in enumerate(results.items()):
        axes[i, 0].imshow(res['original'], cmap='gray')
        axes[i, 0].set_title(name)
        axes[i, 0].axis('off')

        methods = ['global', 'otsu', 'adaptive', 'iterative']
        titles = [
            'Global',
            f'Otsu ({res["T_otsu"]:.0f})',
            'Adaptive',
            f'Iterative ({res["T_iter"]:.0f})'
        ]

        for j, (m, t) in enumerate(zip(methods, titles), 1):
            axes[i, j].imshow(res[m], cmap='gray')
            axes[i, j].set_title(t)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

    # =========================
    # 2. HISTOGRAM (4 gambar)
    # =========================
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (name, res) in enumerate(results.items()):
        hist = cv2.calcHist([res['original']], [0], None, [256], [0, 256])
        axes[i].plot(hist)
        axes[i].axvline(127)
        axes[i].axvline(res['T_otsu'])
        axes[i].axvline(res['T_iter'])
        axes[i].set_title(name)
        axes[i].set_xlim([0, 255])

    plt.tight_layout()
    plt.show()

    # =========================
    # 3. EVALUASI (6 gambar)
    # =========================
    gt = np.zeros((256, 256), dtype=np.uint8)
    gt[30:150, 30:150] = 1
    gt[100:220, 100:220] = 1

    bimodal = results['Bimodal Image']

    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(bimodal['original'], cmap='gray')
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt, cmap='gray')
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')

    methods = ['global', 'otsu', 'adaptive']

    for i, m in enumerate(methods):
        axes[1, i].imshow(bimodal[m], cmap='gray')
        axes[1, i].set_title(m)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


    return results


# =========================
# JALANKAN
# =========================
praktikum_9_1()