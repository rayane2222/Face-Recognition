#!/usr/bin/env python3
import os
import time
import numpy as np
import onnxruntime as ort

import gi
gi.require_version("Gst", "1.0")
gi.require_version("Gtk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gst, Gtk, GLib, GdkPixbuf, Gdk

import cairo

# =========================
# CONFIG
# =========================
APP_NAME = "Face Detection (BlazeFace NPU)"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "BlazeFace_Optimized.onnx")

INPUT_W = 128
INPUT_H = 128

CAP_W = 640
CAP_H = 480

SCORE_THRESH = 0.7
NMS_IOU_THRESH = 0.3
MAX_DETECTIONS = 10

# =========================
# UTILS
# =========================
def resize_nn_rgb(img, out_w, out_h):
    in_h, in_w, _ = img.shape
    ys = (np.arange(out_h) * (in_h / out_h)).astype(np.int32)
    xs = (np.arange(out_w) * (in_w / out_w)).astype(np.int32)
    return img[ys[:, None], xs[None, :], :]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

def nms(boxes, scores):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0 and len(keep) < MAX_DETECTIONS:
        i = idxs[0]
        keep.append(i)
        idxs = idxs[1:]
        idxs = [j for j in idxs if iou(boxes[i], boxes[j]) < NMS_IOU_THRESH]
    return keep

# =========================
# ANCHORS
# =========================
def generate_anchors():
    anchors = []
    for stride, n in [(8,2),(16,6)]:
        fm = INPUT_W // stride
        for y in range(fm):
            for x in range(fm):
                cx = (x+0.5)/fm
                cy = (y+0.5)/fm
                for _ in range(n):
                    anchors.append([cx,cy,1,1])
    return np.array(anchors,dtype=np.float32)

ANCHORS = generate_anchors()

def decode(outputs):
    out0, out1 = outputs
    if out0.shape[-1]==16:
        regs = out0[0]
        scores = sigmoid(out1[0,:,0])
    else:
        regs = out1[0]
        scores = sigmoid(out0[0,:,0])

    dx,dy,dw,dh = regs[:,0],regs[:,1],regs[:,2],regs[:,3]
    ax,ay,aw,ah = ANCHORS[:,0],ANCHORS[:,1],ANCHORS[:,2],ANCHORS[:,3]

    cx = dx/128*aw + ax
    cy = dy/128*ah + ay
    w  = dw/128*aw
    h  = dh/128*ah

    x1 = np.clip(cx-w/2,0,1)
    y1 = np.clip(cy-h/2,0,1)
    x2 = np.clip(cx+w/2,0,1)
    y2 = np.clip(cy+h/2,0,1)

    boxes = np.stack([x1,y1,x2,y2],axis=1)
    return boxes,scores

# =========================
# DRAW BOX
# =========================
def draw_rect(img,x1,y1,x2,y2):
    x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
    img[y1:y1+2,x1:x2]=(255,0,0)
    img[y2-2:y2,x1:x2]=(255,0,0)
    img[y1:y2,x1:x1+2]=(255,0,0)
    img[y1:y2,x2-2:x2]=(255,0,0)

# =========================
# APP
# =========================
class App(Gtk.Window):
    def __init__(self):
        super().__init__(title=APP_NAME)
        self.set_default_size(CAP_W,CAP_H)

        self.image = Gtk.Image()
        self.add(self.image)
        self.connect("destroy",Gtk.main_quit)

        # ONNX Session
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            MODEL_PATH,
            sess_options=so,
            providers=["VSINPUExecutionProvider","CPUExecutionProvider"]
        )

        print("Using providers:", self.sess.get_providers())
        self.input_name = self.sess.get_inputs()[0].name

        # FPS counters
        self.inf_count=0
        self.disp_count=0
        self.t0=time.time()
        self.fps_inf=0
        self.fps_disp=0
        self.last_boxes=[]

        # GStreamer
        Gst.init(None)
        pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw,format=BGR,width={CAP_W},height={CAP_H} ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )

        self.pipeline=Gst.parse_launch(pipeline)
        self.sink=self.pipeline.get_by_name("sink")
        self.sink.connect("new-sample",self.on_frame)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.show_all()

    def update_fps(self):
        dt=time.time()-self.t0
        if dt>=1:
            self.fps_inf=self.inf_count/dt
            self.fps_disp=self.disp_count/dt
            self.inf_count=0
            self.disp_count=0
            self.t0=time.time()

    def on_frame(self,sink):
        sample=sink.emit("pull-sample")
        buf=sample.get_buffer()
        caps=sample.get_caps()
        w=caps.get_structure(0).get_value("width")
        h=caps.get_structure(0).get_value("height")

        ok,mapinfo=buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        frame=np.frombuffer(mapinfo.data,dtype=np.uint8).reshape((h,w,3))
        buf.unmap(mapinfo)

        rgb=frame[:,:,::-1]

        small=resize_nn_rgb(rgb,INPUT_W,INPUT_H)
        inp=(small.astype(np.float32)/255.0)
        inp=np.transpose(inp,(2,0,1))[None,:,:,:]

        outputs=self.sess.run(None,{self.input_name:inp})
        boxes,scores=decode(outputs)

        mask=np.where(scores>=SCORE_THRESH)[0]
        boxes=boxes[mask]
        scores=scores[mask]

        if len(boxes)>0:
            keep=nms(boxes,scores)
            boxes=boxes[keep]

        self.last_boxes=[(
            b[0]*w,b[1]*h,b[2]*w,b[3]*h
        ) for b in boxes]

        self.inf_count+=1

        disp=rgb.copy()
        for b in self.last_boxes:
            draw_rect(disp,*b)

        self.disp_count+=1
        self.update_fps()

        GLib.idle_add(self.update_image,disp)
        return Gst.FlowReturn.OK

    def update_image(self,frame):
        h,w,_=frame.shape

        pixbuf=GdkPixbuf.Pixbuf.new_from_data(
            frame.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,8,w,h,w*3
        )

        surface = Gdk.cairo_surface_create_from_pixbuf(pixbuf,1,None)
        cr = cairo.Context(surface)

        # Fond semi-transparent
        cr.set_source_rgba(0,0,0,0.6)
        cr.rectangle(0,0,270,100)
        cr.fill()

        cr.select_font_face("Sans",cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(22)

        cr.set_source_rgb(0,1,0)
        cr.move_to(15,30)
        cr.show_text(f"INF_FPS: {self.fps_inf:.1f}")

        cr.move_to(15,60)
        cr.show_text(f"DISP_FPS: {self.fps_disp:.1f}")

        cr.set_source_rgb(1,1,0)
        cr.move_to(15,90)
        cr.show_text(f"Faces: {len(self.last_boxes)}")

        cr.stroke()

        self.image.set_from_surface(surface)
        return False

if __name__=="__main__":
    App()
    Gtk.main()
