from pathlib import Path
import json
from ultralytics import RTDETR

DATASET=Path("yolo_kitti_dataset")/"dataset.yaml"
MODEL="rtdetr-l.pt"
PROJECT="runs"
NAME="rtdetr_l_detect"

model=RTDETR(MODEL)
model.train(
    data=str(DATASET),
    epochs=50,
    imgsz=1280,
    batch=4,
    device=0,
    workers=8,
    amp=True,
    optimizer="AdamW",
    lr0=1e-4,
    cos_lr=True,
    patience=20,
    project=PROJECT,
    name=NAME,
    save=True,
    plots=True,
)

run_dir=Path(PROJECT)/NAME
best=run_dir/"weights"/"best.pt"
best_model=RTDETR(str(best))

val=best_model.val(
    data=str(DATASET),
    split="val",
    imgsz=1280,
    batch=1,
    plots=True,
    save_json=True,
)

test=best_model.val(
    data=str(DATASET),
    split="test",
    imgsz=1280,
    batch=1,
    plots=True,
    save_json=True,
)

summary={
 "validation":{
   "precision":float(val.box.mp),
   "recall":float(val.box.mr),
   "map50":float(val.box.map50),
   "map50_95":float(val.box.map)},
 "test":{
   "precision":float(test.box.mp),
   "recall":float(test.box.mr),
   "map50":float(test.box.map50),
   "map50_95":float(test.box.map)}
}
with open(run_dir/"final_metrics.json","w") as f:
    json.dump(summary,f,indent=4)

print("Training complete.")
print("Best:",best)
print("results.csv, results.png, PR/P/R/F1 curves and confusion matrix are saved automatically by Ultralytics.")
