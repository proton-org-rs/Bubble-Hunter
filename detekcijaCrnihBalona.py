import cv2
import numpy as np
import matplotlib.pyplot as plt

Blue = (255, 0, 0)
Green = (0, 255, 0)
Red = (0, 0, 255)
Pink = (255, 0, 255)

def gamma_correction(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def nadjiCrno(img):
    #nzm dal je potreban blur ovde
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    #cv2.imshow("blur", blur)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    #histogram = cv2.calcHist([v], [0], None, [256], [0, 256])
    #plt.plot(histogram)
    #plt.show()

    prag, thresh = cv2.threshold(v, 70, 255, cv2.THRESH_BINARY_INV)
    samoCrno = cv2.bitwise_and(img, img, mask=thresh)
    #cv2.imshow("samo crno", samoCrno)
    cv2.imshow("detektovano", thresh)

    distTransform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    granica, sureForeGround = cv2.threshold(distTransform, 0.3 * distTransform.max(), 255, 0)
    sureForeGround = np.uint8(sureForeGround)
    cv2.imshow("sure_fg", sureForeGround)

    brojPovrsina, markers = cv2.connectedComponents(sureForeGround)
    markers = markers + 1

    unknown = cv2.subtract(thresh, sureForeGround)
    cv2.imshow("unknown", unknown)
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    imgSaGranicama = img.copy()
    imgSaGranicama[markers == -1] = [0, 0, 255]
    cv2.imshow("watershed granice", imgSaGranicama)

    boxes = []
    for markerId in range(2, markers.max() + 1):
        balonMask = np.uint8(markers == markerId) * 255
        contours, hierarchy = cv2.findContours(balonMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity > 0.1:
                boxes.append(cv2.boundingRect(c))

    return boxes


def nadjiKonture(img):
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    #cv2.imshow("blur", blur)

    edges = cv2.Canny(blur, 40, 80)
    cv2.imshow("edges", edges)

    kernel = np.ones((9, 9), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    cv2.imshow("edges_dilated", edges_dilated)

    contours, hierarchy = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity > 0.7:
            boxes.append(cv2.boundingRect(c))

    return boxes

def nadjiKrugove(img):
    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame, (11, 11), 0)

    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                              param1=60, param2=80, minRadius=5, maxRadius=0)
    # param1 je sensitivity, veci broj znaci manje krugova nadje
    # param2 je accuracy, number of edge points needed, veci broj znaci manje krugova
    boxes = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = int(i[0]), int(i[1]), int(i[2])
            height, width = img.shape[:2]

            x1 = np.clip(x - r, 0, width)
            y1 = np.clip(y - r, 0, height)
            x2 = np.clip(x + r, 0, width)
            y2 = np.clip(y + r, 0, height)
            boxes.append((x1, y1, 2*r, 2*r))
    return boxes

def povrsina(box):
    return box[2] * box[3]

def presek(boxA, boxB):
    xA1, yA1, wA, hA = boxA
    xA2, yA2 = xA1 + wA, yA1 + hA
    xB1, yB1, wB, hB = boxB
    xB2, yB2 = xB1 + wB, yB1 + hB

    interX1 = max(xA1, xB1)
    interY1 = max(yA1, yB1)
    interX2 = min(xA2, xB2)
    interY2 = min(yA2, yB2)
    interW = max(0, interX2 - interX1)
    interH = max(0, interY2 - interY1)

    return (interX1, interY1, interW, interH)


def iou2(boxA, boxB):
    interX1, interY1, interW, interH = presek(boxA, boxB)
    interArea = interW * interH

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    unionArea = areaA + areaB - interArea

    if unionArea == 0:
        return 0
    return interArea / unionArea

def iou3(boxA, boxB, boxC):
    boxAB = presek(boxA, boxB)
    return iou2(boxAB, boxC)

def spojiDetekcije(boxes1, boxes2, boxes3):
    potvrdjeni = []
    #za presek sva 3
    # for boxA in boxes1:
    #     for boxB in boxes2:
    #         if iou2(boxA, boxB) < 0.3:
    #             break
    #         for boxC in boxes3:
    #             if iou3(boxA, boxB, boxC) > 0.2:
    #                 potvrdjeni.append(boxB)
    #                 break

    #za A and (B or C)
    for boxA in boxes1:
        nadjeno = False
        for boxB in boxes2:
            if iou2(boxA, boxB) > 0.5:
                potvrdjeni.append(min(boxA, boxB, key=povrsina))
                nadjeno = True
                break
        if not nadjeno:
            for boxC in boxes3:
                if iou2(boxA, boxC) > 0.5:
                    potvrdjeni.append(min(boxA, boxC, key=povrsina))
                    break

    return potvrdjeni


def putBoundingBox(img, boxes, color):
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, img = cap.read()
    if not ret:
        break
    corrected = gamma_correction(img, gamma=1)
    cv2.imshow("corrected", corrected)

    hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge([h, s, v_clahe])
    bgr_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    cv2.imshow("clahe", bgr_clahe)

    boxes1 = nadjiCrno(bgr_clahe)
    boxes2 = nadjiKonture(bgr_clahe)
    boxes3 = nadjiKrugove(bgr_clahe)

    potvrdjeneDetekcije = spojiDetekcije(boxes1, boxes2, boxes3)

    img_prve = img.copy()
    putBoundingBox(img_prve, boxes1, Blue)
    cv2.imshow("prve", img_prve)

    img_druge = img.copy()
    putBoundingBox(img_druge, boxes2, Red)
    cv2.imshow("druge", img_druge)

    img_trece = img.copy()
    putBoundingBox(img_trece, boxes3, Pink)
    cv2.imshow("trece", img_trece)

    img_rezultat = img.copy()
    putBoundingBox(img_rezultat, potvrdjeneDetekcije, Green)
    cv2.imshow("rezultat", img_rezultat)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()