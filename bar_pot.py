import cv2
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

VIDEO_PATH = 'edited_pot.mp4'
OUTPUT_PATH = 'final_final_6th_attempt.mp4'
ALERT_THRESHOLD_SECONDS = 60

model = YOLO('yolov8x.pt')  # Use yolov11x.pt if you want to modify

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 240:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

person_first_seen = {}  # {id: (center, first_seen_time)}
total_time = defaultdict(float)  # {id: total_seconds}
last_seen_time = {}  # {id: last_detection_time}
next_person_id = 0

def is_person_on_truck(person_box, truck_box):
    px1, py1, px2, py2 = person_box
    tx1, ty1, tx2, ty2 = truck_box

    truck_top = ty1
    truck_bottom = ty1 + int((ty2 - ty1) * 0.3)
    truck_left = tx1
    truck_right = tx2

    ix1 = max(px1, truck_left)
    iy1 = max(py1, truck_top)
    ix2 = min(px2, truck_right)
    iy2 = min(py2, truck_bottom)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection_area = iw * ih

    person_area = (px2 - px1) * (py2 - py1)
    if person_area == 0:
        return False

    overlap_ratio = intersection_area / person_area
    return overlap_ratio > 0.3

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    persons = [boxes[i] for i, c in enumerate(classes) if c == 0 and scores[i] > 0.5]
    trucks = [boxes[i] for i, c in enumerate(classes) if c == 7 and scores[i] > 0.5]

    current_persons_on_truck = []
    for person_box in persons:
        for truck_box in trucks:
            if is_person_on_truck(person_box, truck_box):
                current_persons_on_truck.append(person_box)
                break

    new_person_last_seen = {}
    active_ids = []
    for person_box in current_persons_on_truck:
        center = get_center(person_box)
        matched_id = None

        for pid, (last_center, _) in person_first_seen.items():
            if abs(center[0] - last_center[0]) < 30 and abs(center[1] - last_center[1]) < 30:
                matched_id = pid
                break

        if matched_id is None:
            matched_id = next_person_id
            next_person_id += 1
            person_first_seen[matched_id] = (center, current_time)

        # Update time accounting for detection gaps
        if matched_id in last_seen_time:
            time_diff = current_time - last_seen_time[matched_id]
            if time_diff < 2:  # Allow up to 2 seconds gap
                total_time[matched_id] += time_diff
        else:
            total_time[matched_id] += 0  # First detection, no time to add

        last_seen_time[matched_id] = current_time
        new_person_last_seen[matched_id] = (center, person_first_seen[matched_id][1])
        active_ids.append(matched_id)

        # Draw bounding box, label, and timer
        x1, y1, x2, y2 = map(int, person_box)
        timer = int(total_time[matched_id])
        color = (0, 255, 0) if timer < ALERT_THRESHOLD_SECONDS else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = "PERSON_ON_TRUCK"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - label_height - 20), (x1 + label_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        timer_text = f"{timer}s"
        cv2.putText(frame, timer_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Remove stale entries (people who left the truck)
    for pid in list(last_seen_time.keys()):
        if pid not in active_ids:
            del last_seen_time[pid]

    out.write(frame)
    cv2.imshow("Person-on-Truck Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

'''
# Generate bar chart report
if total_time:
    plt.figure(figsize=(10, 6))
    ids = list(total_time.keys())
    times = [total_time[id] for id in ids]
    bars = plt.bar([f'{id}' for id in ids], times, color='skyblue')
    plt.xlabel('Person')
    plt.ylabel('Total Time on Truck (seconds)')
    plt.title('Total Time Spent on Truck per Person')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}s', ha='center', va='bottom')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('time_report.png')
    plt.close()
    print("Report generated: time_report.png")
else:
    print("No persons detected on trucks")

print(f"Detection video saved as: {OUTPUT_PATH}")



# Generate bar chart report
if total_time:
    plt.figure(figsize=(10, 6))
    ids = list(total_time.keys())
    times = [total_time[id] for id in ids]
    bars = plt.bar([f'{id}' for id in ids], times, color='skyblue')
    plt.xlabel('Person')
    plt.ylabel('Total Time on Truck (seconds)')
    plt.title('Total Time Spent on Truck per Person')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}s', ha='center', va='bottom')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Add total sum annotation in dark red, larger font, top-right corner
    total_sum = sum(times)
    plt.text(
        0.98, 0.95,  # x=98% (right), y=95% (top) in axes fraction coordinates
        f'Total Time in seconds: {total_sum:.1f}s',
        ha='right', va='top',
        fontsize=16, fontweight='bold',
        color='darkred',
        transform=plt.gca().transAxes
    )

    plt.savefig('time_report.png')
    plt.close()
    print("Report generated: time_report.png")
else:
    print("No persons detected on trucks")
    '''

import matplotlib.pyplot as plt
from collections import defaultdict

# Example data (replace this with your actual 'total_time' dictionary)
total_time = defaultdict(float, {0: 120.5, 1: 75.3, 2: 45.0})

# Generate bar chart report
if total_time:
    plt.figure(figsize=(10, 6))
    ids = list(total_time.keys())
    times = [total_time[id] for id in ids]
    bars = plt.bar([f'{id}' for id in ids], times, color='skyblue')
    plt.xlabel('Person')
    plt.ylabel('Total Time on Truck (seconds)')
    plt.title('Total Time Spent on Truck per Person')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}s', ha='center', va='bottom')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    total_sum = sum(times)
    total_minutes = total_sum / 60

    # Total time in seconds 
    plt.text(
        0.98, 0.95,
        f'Total Time in seconds: {total_sum:.1f}s',
        ha='right', va='top',
        fontsize=16, fontweight='bold',
        color='darkred',
        transform=plt.gca().transAxes
    )
    # Total time in minutes 
    plt.text(
        0.98, 0.90,
        f'Total Time in minutes: {total_minutes:.2f} min',
        ha='right', va='top',
        fontsize=16, fontweight='bold',
        color='darkred',
        transform=plt.gca().transAxes
    )

    plt.savefig('time_report3.png')
    plt.close()
    print("Report generated: time_report.png")
else:
    print("No persons detected on trucks")


