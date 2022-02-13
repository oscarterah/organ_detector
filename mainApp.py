from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
import random
import colorsys
from skimage.measure import find_contours
from kivymd.app import MDApp
import numpy as np
from kivy.config import Config
Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '600')
Config.set('graphics', 'resizable', False)
Config.write()
from kivy.core.text import LabelBase
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.metrics import dp
from kivy.core.window import Window
import time
LabelBase.register(name="terah", fn_regular= "Arial Black.ttf")


CLASS_NAMES = ['BG', 'liver']

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)

def random_colors(N):
    brightness = 1.0
    hsv = [(i/N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


colors = random_colors(len(CLASS_NAMES))
class_dict = {
    name: color for name, color in zip(CLASS_NAMES, colors)
}


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c * 255,
            image[:, :, n]
        )
    return image

def conts(image, color, mask):
    padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1]+2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        verts = np.fliplr(verts) - 1
        verts = np.array(verts, np.int32)
        verts = verts.reshape(-1,1,2)
        image = cv2.polylines(image, verts, True, color)

    return image

def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
#        color = class_dict[label]
        color = (225,0,0)
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        image = conts(image,color,mask)
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def load_image():
    img = 'ty1.jpg'
    return img


def save_image():
    model = mrcnn.model.MaskRCNN(mode="inference", config=SimpleConfig(),model_dir=os.getcwd())
    model.load_weights(filepath="mask_rcnn.h5", by_name=True)
    image = cv2.imread(load_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r = model.detect([image], verbose=0)
    r = r[0]
    img = display_instances(image, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES, r['scores'])
    return img

class BckBtn(FloatLayout):
    def __init__(self, mainwid, **kwargs):
        super().__init__()
        self.mainwid = mainwid 

    def bck_start(self):
        self.mainwid.goto_startpoint()

class Load(FloatLayout):
    def __init__(self, mainwid, **kwargs):
        super().__init__()
        self.mainwid = mainwid

    def image_load(self):
        self.mainwid.Loadm()

class Detect(FloatLayout):
    def __init__(self, mainwid, **kwargs):
        super().__init__()
        self.mainwid = mainwid

    def make_detection(self):
        save_image()
        time.sleep(1)
        self.mainwid.detectm()


class Zoom(FloatLayout):
    def __init__(self,mainwid, **Kwargs):
        super().__init__()
        self.mainwid = mainwid

    def zoom(self):
        self.mainwid.zoom_in()
    
        

class StartWid(Screen):
    def __init__(self, **kwargs):
        super().__init__()
        self.detect = Detect(self)
        self.load = Load(self)
        self.zoom = Zoom(self)
        self.add_widget(self.zoom)
        self.add_widget(self.detect)
        self.add_widget(self.load)
        Window.bind(on_dropfile=self._on_file_drop)

    def _on_file_drop(self, window, file_path):
        print(file_path)
        
    def create_wid(self, img):
        self.img = img
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = cv2.flip(self.img, 0)
        w, h, _ = self.img.shape
        texture = Texture.create(size=(w, h))
        texture.blit_buffer(self.img.flatten(), colorfmt='rgb', bufferfmt='ubyte')
        self.w_img = Image(size=(w, h), texture=texture)
        self.w_img.keep_ratio = False
        self.w_img.allow_stretch = True
        self.w_img.size_hint_x = 0.75
        self.w_img.size_hint_y = 0.75
        self.w_img.pos = (31.4, 100)
        self.add_widget(self.w_img)

    def Loadm(self): 
        self.img = cv2.imread(load_image(), cv2.IMREAD_UNCHANGED)
        self.create_wid(self.img)

    def detectm(self):
        self.img = save_image()
        self.create_wid(self.img)

    def zoom_in(self):
        self.img = cv2.imread(load_image(), cv2.IMREAD_UNCHANGED)
        self.img = cv2.resize(self.img, None, fx=2, fy=2)
        self.create_wid(self.img)
        print('zooming')

class MainApp(MDApp):
    title = 'MASK_LAP'

    def build(self):
        return StartWid()
