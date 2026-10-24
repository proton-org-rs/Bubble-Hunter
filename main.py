import cv2
from detekcijaCrnihBalona import gammaCorrection, nadjiCrno, nadjiKonture, nadjiKrugove, spojiDetekcije, fuzijaDetekcija, putBoundingBox
from tracker import azurirajTrackove
from konverzija import angle_calculation

Blue = (255, 0, 0)
Green = (0, 255, 0)
Red = (0, 0, 255)
Pink = (255, 0, 255)

d = 50 #cm

trackovi = []

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, img = cap.read()
    if not ret:
        break

    visina = img.shape[0]
    sirina = img.shape[1]

    corrected = gammaCorrection(img, gamma=1)
    #cv2.imshow("corrected", corrected)

    hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    vClahe = clahe.apply(v)
    hsvClahe = cv2.merge([h, s, vClahe])
    bgrClahe = cv2.cvtColor(hsvClahe, cv2.COLOR_HSV2BGR)
    cv2.imshow("clahe", bgrClahe)

    boxes1 = nadjiCrno(bgrClahe)
    boxes2 = nadjiKonture(bgrClahe)
    boxes3 = nadjiKrugove(bgrClahe)

    #potvrdjeneDetekcije = spojiDetekcije(boxes1, boxes2, boxes3)
    potvrdjeneDetekcije = fuzijaDetekcija(boxes1, boxes2, boxes3)

    imgPrve = img.copy()
    putBoundingBox(imgPrve, boxes1, Blue)
    cv2.imshow("prve", imgPrve)

    imgDruge = img.copy()
    putBoundingBox(imgDruge, boxes2, Red)
    cv2.imshow("druge", imgDruge)

    imgTrece = img.copy()
    putBoundingBox(imgTrece, boxes3, Pink)
    cv2.imshow("trece", imgTrece)

    imgDetekcije = img.copy()
    putBoundingBox(imgDetekcije, potvrdjeneDetekcije, Green)
    cv2.imshow("detekcije", imgDetekcije)

    trackovi = azurirajTrackove(trackovi, potvrdjeneDetekcije)

    for track in trackovi:
        if not track["potvrdjen"]:
            continue
        x, y, w, h = track["smoothBox"]

        angle1 = angle_calculation(x + w/2, d, sirina/2)  # levo/desno
        angle2 = angle_calculation(y + h/2, d, visina/2)  # gore/dole

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"ID {track['potvrdjenID']}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("rezultat", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()