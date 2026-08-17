from detekcijaCrnihBalona import iou2

nextId = 0
confirmedID = 0

def napraviTrack(box):
    global nextId
    track = {
        "id": nextId,
        "box": box,
        "uzastopniPogoci": 1,
        "uzastopniPromasaji": 0,
        "potvrdjen": False,
        "potvrdjenID": None
    }
    nextId += 1
    return track

def azurirajTrackove(trackovi, noveDetekcije, iouPrag=0.3, pragPotvrde=3, pragBrisanja=5):
    upareniTrackovi = set()
    upareneDetekcije = set()

    # 1. korak logika i updatuje polozaj uparenih
    for i, track in enumerate(trackovi):
        najboljiIou = iouPrag
        najboljiJ = -1
        for j, det in enumerate(noveDetekcije):
            if j in upareneDetekcije:
                continue
            trenutniIou = iou2(track["box"], det)
            if trenutniIou > najboljiIou:
                najboljiIou = trenutniIou
                najboljiJ = j

        if najboljiJ != -1:
            track["box"] = noveDetekcije[najboljiJ]
            track["uzastopniPogoci"] += 1
            track["uzastopniPromasaji"] = 0

            upareniTrackovi.add(i) # za 2. korak
            upareneDetekcije.add(najboljiJ) # za 3. korak

    # 2. korak zapisuje promasaje za trackove
    for i, track in enumerate(trackovi):
        if i not in upareniTrackovi:
            track["uzastopniPogoci"] = 0
            track["uzastopniPromasaji"] += 1

    # 3. korak pravi nove trackove za neuparene detekcije
    for j, det in enumerate(noveDetekcije):
        if j not in upareneDetekcije:
            trackovi.append(napraviTrack(det))

    # 4. korak potvrdjuje trackove
    global confirmedID
    for track in trackovi:
        if track["uzastopniPogoci"] >= pragPotvrde:
            if not track["potvrdjen"]:
                confirmedID += 1
                track["potvrdjenID"] = confirmedID

            track["potvrdjen"] = True

    # 5. korak brise stare trackove
    trackovi = [t for t in trackovi if t["uzastopniPromasaji"] < pragBrisanja]

    return trackovi