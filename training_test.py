from ultralytics import YOLO


if __name__ == "__main__":
    # model = YOLO('runs/detect/train3/weights/best.pt')
    # model = YOLO("yolov8n.pt")
    model = YOLO("yolo11n.pt")
    #model.to("cuda")
    #model.train(data="data_metadata.yaml", epochs=100, batch=32, translate=0.4, scale=0.2, mixup=0.3)
    model.train(data="data_metadata.yaml", epochs=30, batch=32, device="cuda")# multi_scale=True

"""
task=detect, mode=train, model=yolo11n.pt, data=data_metadata.yaml, epochs=30,
time=None, patience=100, batch=32, imgsz=640, save=True, save_period=-1, cache=False,
device=cuda, workers=8, project=None, name=train4, exist_ok=False, pretrained=True,
optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False,
cos_lr=False, close_mosaic=10, resume=False, amp=False, fraction=1.0, profile=False,
freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True,
split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False,
dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False,
augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, 
save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, 
show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, 
int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, 
momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, 
box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, 
translate=0.6, scale=0.2, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, 
mixup=0.4, copy_paste=0.5, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, 
cfg=None, tracker=botsort.yaml, save_dir=runs\\detect\\train4
Overriding model.yaml nc=80 with nc=26
"""