import numpy as np
import cv2
import matplotlib.pyplot as plt

def praktikum_9_2():
    print("\nPRAKTIKUM 9.2: EDGE DETECTION DAN REGION-BASED SEGMENTATION")
    print("=" * 70)

    # =========================
    # BUAT GAMBAR TEST
    # =========================
    def create_edge_test_image():
        img = np.zeros((300, 400), dtype=np.uint8)

        cv2.rectangle(img, (50, 50), (150, 150), 100, -1)
        cv2.rectangle(img, (151, 50), (250, 150), 200, -1)

        for i in range(50, 150):
            img[160:240, i] = 50 + (i - 50) * 2

        triangle_cnt = np.array([(300, 160), (350, 240), (250, 240)])
        cv2.drawContours(img, [triangle_cnt], 0, 150, -1)

        cv2.line(img, (50, 260), (350, 260), 200, 3)

        noise = np.random.normal(0, 15, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img

    # =========================
    # EDGE DETECTION
    # =========================
    def sobel(image):
        sx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        sy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        mag = np.sqrt(sx**2 + sy**2)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return mag.astype(np.uint8)

    def prewitt(image):
        kx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        ky = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        px = cv2.filter2D(image.astype(float), -1, kx)
        py = cv2.filter2D(image.astype(float), -1, ky)
        mag = np.sqrt(px**2 + py**2)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return mag.astype(np.uint8)

    def canny(image):
        blur = cv2.GaussianBlur(image, (5,5), 1.4)
        return cv2.Canny(blur, 50, 150)

    def laplacian(image):
        lap = cv2.Laplacian(image, cv2.CV_64F)
        lap = np.abs(lap)
        lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
        return lap.astype(np.uint8)

    # =========================
    # REGION GROWING (AMAN)
    # =========================
    def region_growing(image, seeds, threshold=25):
        h, w = image.shape
        visited = np.zeros((h, w), dtype=bool)
        result = np.zeros_like(image)

        for seed in seeds:
            stack = [seed]

            while stack:
                x, y = stack.pop()

                if not (0 <= x < h and 0 <= y < w):
                    continue
                if visited[x, y]:
                    continue

                visited[x, y] = True
                result[x, y] = 255

                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if not visited[nx, ny]:
                            if abs(int(image[nx, ny]) - int(image[x, y])) < threshold:
                                stack.append((nx, ny))

        return result

    # =========================
    # PROSES
    # =========================
    img = create_edge_test_image()

    sob = sobel(img)
    pre = prewitt(img)
    can = canny(img)
    lap = laplacian(img)

    seeds = [(75,100),(200,100),(250,200)]
    rg = region_growing(img, seeds)

    # =========================
    # 1. EDGE DETECTION (6 GAMBAR)
    # =========================
    fig1, ax = plt.subplots(2,3, figsize=(15,10))

    ax[0,0].imshow(img, cmap='gray')
    ax[0,0].set_title("Original")

    ax[0,1].imshow(sob, cmap='gray')
    ax[0,1].set_title("Sobel")

    ax[0,2].imshow(pre, cmap='gray')
    ax[0,2].set_title("Prewitt")

    ax[1,0].imshow(can, cmap='gray')
    ax[1,0].set_title("Canny")

    ax[1,1].imshow(lap, cmap='gray')
    ax[1,1].set_title("Laplacian")

    combined = np.maximum(sob, can)
    ax[1,2].imshow(combined, cmap='gray')
    ax[1,2].set_title("Combined")

    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show(block=True)

    # =========================
    # 2. REGION SEGMENTATION (6 GAMBAR)
    # =========================
    fig2, ax = plt.subplots(2,3, figsize=(15,10))

    ax[0,0].imshow(img, cmap='gray')
    ax[0,0].set_title("Original")

    ax[0,1].imshow(rg, cmap='gray')
    ax[0,1].set_title("Region Growing")

    seed_img = img.copy()
    for s in seeds:
        cv2.circle(seed_img, (s[1], s[0]), 5, 255, -1)

    ax[0,2].imshow(seed_img, cmap='gray')
    ax[0,2].set_title("Seed Points")

    ax[1,0].imshow(cv2.Canny(rg,50,150), cmap='gray')
    ax[1,0].set_title("Region Boundary")

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[rg>0] = [0,255,0]
    ax[1,1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax[1,1].set_title("Overlay")

    ax[1,2].imshow(np.abs(img - rg), cmap='gray')
    ax[1,2].set_title("Difference")

    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show(block=True)


    return img, sob, pre, can, lap, rg


# =========================
# JALANKAN
# =========================
praktikum_9_2()