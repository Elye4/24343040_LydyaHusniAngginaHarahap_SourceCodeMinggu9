import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# =========================
# 1. BUAT DATASET CITRA (TANPA FILE)
# =========================
def create_images():
    images = {}

    # Bimodal
    img1 = np.zeros((256,256), dtype=np.uint8)
    cv2.rectangle(img1,(50,50),(200,200),200,-1)
    images['Bimodal'] = img1

    # Iluminasi tidak merata
    img2 = np.zeros((256,256), dtype=np.uint8)
    for i in range(256):
        img2[:,i] = i
    cv2.circle(img2,(128,128),50,255,-1)
    images['Uneven'] = img2

    # Overlapping
    img3 = np.zeros((256,256), dtype=np.uint8)
    cv2.circle(img3,(100,120),60,180,-1)
    cv2.circle(img3,(150,120),60,120,-1)
    images['Overlap'] = img3

    return images

images = create_images()

# =========================
# 2. TAMPILKAN DATASET
# =========================
plt.figure(figsize=(10,4))
for i,(name,img) in enumerate(images.items()):
    plt.subplot(1,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(name)
    plt.axis('off')
plt.suptitle("Dataset Citra")
plt.show()

# =========================
# 3. THRESHOLDING
# =========================
def thresholding(img):
    _, global_t = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mean = cv2.adaptiveThreshold(img,255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,11,2)
    gauss = cv2.adaptiveThreshold(img,255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,11,2)
    return global_t, otsu, mean, gauss

# =========================
# 4. EDGE DETECTION
# =========================
def edge_detection(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    sobel = cv2.magnitude(sobelx, sobely)

    prewittx = cv2.filter2D(img,-1,np.array([[1,0,-1],[1,0,-1],[1,0,-1]]))
    prewitty = cv2.filter2D(img,-1,np.array([[1,1,1],[0,0,0],[-1,-1,-1]]))
    prewitt = np.sqrt(prewittx**2 + prewitty**2)

    canny1 = cv2.Canny(img,50,150)
    canny2 = cv2.Canny(img,100,200)

    return sobel, prewitt, canny1, canny2

# =========================
# 5. REGION GROWING
# =========================
def region_growing(img, seed, threshold=10):
    h, w = img.shape
    visited = np.zeros((h,w), dtype=bool)
    result = np.zeros_like(img)

    stack = [seed]

    while stack:
        x,y = stack.pop()

        if not (0<=x<h and 0<=y<w):
            continue
        if visited[x,y]:
            continue

        visited[x,y] = True
        result[x,y] = 255

        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny = x+dx, y+dy
            if 0<=nx<h and 0<=ny<w:
                if not visited[nx,ny]:
                    if abs(int(img[nx,ny]) - int(img[x,y])) < threshold:
                        stack.append((nx,ny))

    return result

# =========================
# 6. WATERSHED
# =========================
def watershed_seg(img):
    ret, thresh = cv2.threshold(img,0,255,
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist,0.5*dist.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(opening, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    img_color[markers == -1] = [255,0,0]

    return img_color

# =========================
# 7. CONNECTED COMPONENT
# =========================
def connected_comp(img):
    ret, thresh = cv2.threshold(img,0,255,
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    num_labels, labels = cv2.connectedComponents(thresh)
    return labels

# =========================
# 8. PROSES SEMUA
# =========================
for name, img in images.items():
    print("\n=====", name, "=====")
    start = time.time()

    g, o, m, ga = thresholding(img)
    s, p, c1, c2 = edge_detection(img)
    rg = region_growing(img, (100,100))
    w = watershed_seg(img)
    cc = connected_comp(img)

    end = time.time()
    print("Waktu komputasi:", round(end-start,4), "detik")

    # =========================
    # VISUALISASI
    # =========================
    plt.figure(figsize=(12,10))
    plt.suptitle(name)

    plt.subplot(3,3,1); plt.imshow(img,cmap='gray'); plt.title("Original")
    plt.subplot(3,3,2); plt.imshow(g,cmap='gray'); plt.title("Global")
    plt.subplot(3,3,3); plt.imshow(o,cmap='gray'); plt.title("Otsu")

    plt.subplot(3,3,4); plt.imshow(m,cmap='gray'); plt.title("Adaptive Mean")
    plt.subplot(3,3,5); plt.imshow(ga,cmap='gray'); plt.title("Adaptive Gauss")

    plt.subplot(3,3,6); plt.imshow(s,cmap='gray'); plt.title("Sobel")
    plt.subplot(3,3,7); plt.imshow(p,cmap='gray'); plt.title("Prewitt")
    plt.subplot(3,3,8); plt.imshow(c1,cmap='gray'); plt.title("Canny")

    plt.subplot(3,3,9); plt.imshow(w); plt.title("Watershed")

    for i in range(1,10):
        plt.subplot(3,3,i).axis('off')

    plt.tight_layout()
    plt.show()