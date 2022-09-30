import cv2
import threading
import queue
import time
import numpy as np
from .utils import grab_image
from time import sleep


class video_camera(object):
	def __init__(self, src=0):
		self.video = cv2.VideoCapture(src)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		is_true, frame = self.video.read()
		if frame is None:
			return None
		frame_flip = cv2.flip(frame,1)
		# ret, jpg = cv2.imencode('.jpg', frame_flip)
		return frame_flip

class ip_webcam(object):
	def __init__(self, url_):
		self.url = url_

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self, size=(640, 480)):
		img = grab_image(url=self.url)
		if img is None:
			return None
		resized = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR) 
		frame_flip = cv2.flip(resized,1)
		# ret, jpg = cv2.imencode('.jpg', frame_flip)
		return frame_flip

class live_cam(object):
	def __init__(self, url_):
		self.url = cv2.VideoCapture(url_)

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self, size=(640, 480)):
		is_true, frame = self.url.read()
		if frame is None:
			return None
		resized = cv2.resize(frame, size, interpolation = cv2.INTER_LINEAR) 
		# ret, jpg = cv2.imencode('.jpg', resized)
		return resized

class ThreadingCam(object):
	def __init__(self, cam=None, size=(640, 480), max_fps=60):
		self.cam = cam
		self.q = queue.Queue()
		self.max_fps = max_fps
		self.size = size
		self.stop_thread = threading.Event()
		self.event = threading.Event()
		self.event.set()
		self.thing_done= threading.Event()
		self.thing_done.set()
		self.lock = threading.Lock()
		self.start()

	def start(self):
		self.stop_thread.clear()
		t = threading.Thread(target=self._get_frame)
		t.daemon = True
		t.start()

	def stop(self):
		self.stop_thread.set()

	def pause(self):
		self.event.clear()
		self.thing_done.wait()

	def resume(self):
		self.event.set()

	def set_cam(self, cam):
		self.pause()
		with self.lock:
			self.cam = cam
		self.resume()

	def _get_frame(self):
		while True:
			if self.stop_thread.is_set():
				break
			self.event.wait()
			try:
				self.thing_done.clear()
				t1 = time.process_time()
				if self.cam is None:
					frame = None
				else:
					frame = self.cam.get_frame(self.size)
				if not self.q.empty():
					try:
						self.q.get_nowait()
					except queue.Empty:
						pass
				self.q.put(frame)
				t = 1/self.max_fps - (time.process_time() - t1)
				if t > 0:
					time.sleep(t)
			finally:
				self.thing_done.set()

	def get_frame(self):
		return self.q.get()

class backgroundThread(object):
	def __init__(self, people_counter):
		self.people_counter = people_counter
		self.frame = queue.Queue()
		self.frame_draw = queue.Queue()
		self.frame_data = queue.Queue()
		self.stop_thread = threading.Event()
		self.event = threading.Event()
		self.event.set()
		self.thing_done= threading.Event()
		self.thing_done.set()
		self.lock = threading.Lock()
		self.start()

	def start(self):
		self.stop_thread.clear()
		t = threading.Thread(target=self._update)
		t.daemon = True
		t.start()

	def stop(self):
		self.stop_thread.set()

	def pause(self):
		self.event.clear()
		self.thing_done.wait()

	def resume(self):
		self.event.set()

	def set_people_counter(self, people_counter):
		self.pause()
		with self.lock:
			self.people_counter = people_counter
		self.resume()

	def _update(self):
		while True:
			if self.stop_thread.is_set():
				break
			self.event.wait()
			try:
				self.thing_done.clear()
				self.people_counter.update()
				if not self.frame.empty():
					try:
						self.frame.get_nowait()
					except queue.Empty:
						pass
				if not self.frame_draw.empty():
					try:
						self.frame_draw.get_nowait()
					except queue.Empty:
						pass
				if not self.frame_data.empty():
					try:
						self.frame_data.get_nowait()
					except queue.Empty:
						pass
				self.frame_draw.put(self.people_counter.draw())
				self.frame.put(self.people_counter.frame)
				if self.people_counter.frame is None:
					self.frame_data.put(None)
				else:
					h, w = self.people_counter.frame.shape[:2]
					self.frame_data.put({
						'bbox_identities': [{
							'detection_box': [bbox_identity[0]/w, bbox_identity[1]/h, bbox_identity[2]/w, bbox_identity[3]/h],
							'track_id': bbox_identity[4],
							'class_id': bbox_identity[5],
						} for bbox_identity in self.people_counter.bbox_identities],
						'counter': {
							'count': self.people_counter.count_,
							'up': self.people_counter.p_up,
							'down': self.people_counter.p_down,
						}
					})
			finally:
				self.thing_done.set()

	def get(self):
		return self.frame.get(), self.frame_draw.get(), self.frame_data.get()