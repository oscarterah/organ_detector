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
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.scatter import Scatter
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




class Crop(FloatLayout):
    def __init__(self, mainwid, **kwargs):
        super().__init__()
        self.mainwid = mainwid 

    def make_crop(self):
        self.mainwid.startwid.cropm()

class Load(FloatLayout):
    def __init__(self, mainwid, **kwargs):
        super().__init__()
        self.mainwid = mainwid

    def image_load(self):
        self.mainwid.startwid.Loadm()

class Detect(FloatLayout):
    def __init__(self, mainwid, **kwargs):
        super().__init__()
        self.mainwid = mainwid

    def make_detection(self):
        self.mainwid.startwid.detectm()


class Zoom(FloatLayout):
    def __init__(self,mainwid, **Kwargs):
        super().__init__()
        self.mainwid = mainwid

    def zoom(self):
        self.mainwid.startwid.zoom_in()
    
class Content(BoxLayout):
    pass      

class StartWid(ScatterLayout):
    dialog = None  
    def __init__(self,mainwid, **kwargs):
        super().__init__()
        self.mainwid = mainwid
        Window.bind(on_dropfile=self._on_file_drop)

        self.W = False
        self.path = ''
        self.zoomswitch = False 
     

    def _on_file_drop(self, window, file_path):
        self.W = True
        path = file_path.decode("utf-8")
        self.path = path
        self.Loadm()
        
    def load_image(self):
        if(self.W):
            img=self.path
        else: 
            img = 'ty1.jpg'
        return img

    def save_image(self):
        model = mrcnn.model.MaskRCNN(mode="inference", config=SimpleConfig(),model_dir=os.getcwd())
        model.load_weights(filepath="mask_rcnn.h5", by_name=True)
        image = cv2.imread(self.load_image(), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r = model.detect([image], verbose=0)
        r = r[0]
        img = display_instances(image, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES, r['scores'])
        return img

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
        self.img = cv2.imread(self.load_image(), cv2.IMREAD_UNCHANGED)
        self.create_wid(self.img)

    def detectm(self):
        self.img = self.save_image()
        self.create_wid(self.img)


    def on_touch_down(self, touch):
        if(self.zoomswitch):
            x, y = touch.x, touch.y
            self.prev_x = touch.x
            self.prev_y = touch.y

            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    print('down')
                    ## zoom in
                    if self.scale < 10:
                        self.scale = self.scale * 1.1

                elif touch.button == 'scrollup':
                    print('up')  ## zoom out
                    if self.scale > 1:
                        self.scale = self.scale * 0.9

            # if the touch isn't on the widget we do nothing
            if not self.do_collide_after_children:
                if not self.collide_point(x, y):
                    return False

            if 'multitouch_sim' in touch.profile:
                touch.multitouch_sim = True
            # grab the touch so we get all it later move events for sure
            self._bring_to_front(touch)
            touch.grab(self)
            self._touches.append(touch)
            self._last_touch_pos[touch] = touch.pos

    def zoom_in(self):
        self.zoomswitch = True

    def cropm(self):
        self.show_confirmation_dialog()
        self.img = cv2.imread(self.load_image(), cv2.IMREAD_UNCHANGED)
        self.r = cv2.selectROI(self.img)
        self.imgcrop = self.img[int(self.r[1]):int(self.r[1]+self.r[3]), int(self.r[0]):int(self.r[0]+self.r[2])]
        cv2.imshow("image", self.imgcrop)
        cv2.waitKey(0)
        
    def show_confirmation_dialog(self):
        if not self.dialog:
            self.dialog = MDDialog(
                title="Crop Name:",
                type="custom",
                content_cls=Content(),
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        theme_text_color="Custom",
                    ),
                    MDFlatButton(
                        text="OK",
                        theme_text_color="Custom",
                    ),
                ],
            )
        self.dialog.open()

class MainWid(Screen):
    def __init__(self, **kwargs):
        super().__init__()
        self.startwid = StartWid(self)

        wid = Screen(name='start')
        wid.add_widget(self.startwid)
        self.add_widget(wid)
        self.detect = Detect(self)
        self.load = Load(self)
        self.zoom = Zoom(self)
        self.crop = Crop(self)
        self.add_widget(self.crop)
        self.add_widget(self.zoom)
        self.add_widget(self.detect)
        self.add_widget(self.load)

    def goto_start(self):
        self.current='start'

                


class MainApp(MDApp):
    title = 'MASK_LAP'

    def build(self):
        return MainWid()
