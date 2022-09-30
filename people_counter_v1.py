import numpy as np
import cv2


def is_equal(original, duplicate):
    if original is None and duplicate is None:
        return True
    elif original is None and duplicate is not None:
        return False
    elif original is not None and duplicate is None:
        return False
    if original.shape == duplicate.shape:
        difference = cv2.subtract(original, duplicate)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            return True
    return False

class PeopleCounterV1(object):
    def __init__(self, camera, object_detector, tracker, line=[(0, 240), (640, 240)], offset=75):
        self.camera = camera
        self.object_detector = object_detector
        self.tracker = tracker
        self.line = line
        self.offset = offset
        self.p_up = 0
        self.p_down = 0
        self.count_ = 0
        self.bbox_identities = []
        self.frame = None
        self.his = {}

    def compute_color_for_id(self, id_):
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        color = [int((p * (id_ ** 2 - id_ + 1)) % 255) for p in palette]
        return tuple(color)

    def update(self):
        frame = self.camera.get_frame()
        if frame is None:
            self.frame = frame
            self.bbox_identities = []
            return []
        if self.frame is not None:
            if is_equal(frame, self.frame):
                return self.bbox_identities
        self.frame = frame.copy()
        det = self.object_detector.inference(src_img=frame, classes=[0])
        if det is None or not len(det):
            self.tracker.increment_ages()
            self.bbox_identities = []
            return []
        bboxes = [i['bbox'] for i in det]
        bbox_tlwh = [[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]] for bbox in bboxes]
        confidences = [i['confidence'] for i in det]
        classes = [i['class'] for i in det]
        classes_num = [i['class_num'] for i in det]
        remove_indices = []
        for i, bbox in enumerate(bbox_tlwh):
            if bbox[2] == 0 or bbox[3] == 0:
                remove_indices.append(i)
        bbox_tlwh = [i for j, i in enumerate(bbox_tlwh) if j not in remove_indices]
        confidences = [i for j, i in enumerate(confidences) if j not in remove_indices]
        classes_num = [i for j, i in enumerate(classes_num) if j not in remove_indices]
        bbox_identities = self.tracker.update(bbox_tlwh=np.array(bbox_tlwh), confidences=confidences, classes=classes_num, ori_img=frame)
        self.bbox_identities = bbox_identities
        # counting
        self.count()
        return bbox_identities

    def count(self):
        for bbox_identity in self.bbox_identities:
            track_id = bbox_identity[4]
            x1, y1, x2, y2 =  bbox_identity[:4]
            p = (int(x1 + (x2-x1)/2), int(y1 + (y2-y1)/2))
            if p[1] in range(self.line[0][1] - self.offset, self.line[0][1]):
                if self.his.get(track_id) == 1:
                    self.p_up += 1
                self.his[track_id] = 0
            elif p[1] in range(self.line[0][1], self.line[0][1] + self.offset):
                if self.his.get(track_id) == 0:
                    self.p_down += 1
                self.his[track_id] = 1
            elif self.his.get(track_id) is not None:
                self.his.pop(track_id)
        self.count_ = self.p_up - self.p_down

    def draw(self):
        if self.frame is None:
            return self.frame
        frame = self.frame
        cv2.line(frame, self.line[0], self.line[1], (0, 255, 255), 2)
        text = "count: {}\nup: {}\ndown: {}".format(self.count_, self.p_up, self.p_down)
        y0, dy = 20, 20
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(frame, line, (0,y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
        for bbox_identitie in self.bbox_identities:
            cv2.rectangle(frame, (int(bbox_identitie[0]),int(bbox_identitie[1])), (int(bbox_identitie[2]),int(bbox_identitie[3])), self.compute_color_for_id(bbox_identitie[4]), 2)
            cv2.putText(frame , str(bbox_identitie[4]), (int(bbox_identitie[2]+10),int(bbox_identitie[3])), 0, 0.5, self.compute_color_for_id(bbox_identitie[4]))
        return frame