import cv2
import numpy as np
from ultralytics import YOLO

def check_label_collision(new_rect, drawn_rects):
    """Checks if new_rect (x1, y1, x2, y2) overlaps with any existing rects."""
    for (rx1, ry1, rx2, ry2) in drawn_rects:
        # Check intersection
        if not (new_rect[2] < rx1 or new_rect[0] > rx2 or new_rect[3] < ry1 or new_rect[1] > ry2):
            return True
    return False

def draw_3d_volumetric_spotlight(img, feet_pt, box_height, drawn_rects, color=(220, 255, 220), name_tag=None):
    overlay = img.copy()
    h_img, w_img, _ = img.shape
    px, py = feet_pt

    depth_scale = (py / h_img)
    rx = int(16 + depth_scale * 14)
    ry = int(6 + depth_scale * 5)
    pillar_height = int(box_height * 1.25)

    # 1. Base Ellipse & 3D Cylinder Mesh
    cv2.ellipse(overlay, (px, py), (rx, ry), 0, 0, 360, color, -1)
    top_y = py - pillar_height

    pts_cylinder = np.array([
        (px - rx, py), (px + rx, py),
        (px + rx, top_y), (px - rx, top_y)
    ], np.int32)
    
    cv2.fillPoly(overlay, [pts_cylinder], color)
    cv2.ellipse(overlay, (px, top_y), (rx, ry), 0, 0, 360, color, -1)

    # 2. Transparency Fade
    alpha_mask = np.zeros((h_img, w_img), dtype=np.float32)
    cv2.fillPoly(alpha_mask, [pts_cylinder], 0.35)
    cv2.ellipse(alpha_mask, (px, py), (rx, ry), 0, 0, 360, 0.50, -1)

    for y_step in range(max(0, top_y), min(h_img, py)):
        fade_factor = (y_step - top_y) / float(pillar_height)
        alpha_mask[y_step, :] *= (fade_factor ** 1.5)

    alpha_3channel = cv2.merge([alpha_mask, alpha_mask, alpha_mask])
    blended = img.astype(np.float32) * (1.0 - alpha_3channel) + overlay.astype(np.float32) * alpha_3channel
    img[:] = np.clip(blended, 0, 255).astype(np.uint8)

    # 3. Collision-Aware Name Banner Placement
    if name_tag:
        (tw, th), _ = cv2.getTextSize(name_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(18, top_y - 8)
        
        # Test candidate offsets to resolve overlapping labels
        y_offsets = [0, -22, -44, 22, 44]
        best_p1, best_p2, best_y = None, None, tag_y

        for offset in y_offsets:
            cand_y = tag_y + offset
            p1 = (px - int(tw/2) - 6, cand_y - th - 4)
            p2 = (px + int(tw/2) + 6, cand_y + 4)
            cand_rect = (p1[0], p1[1], p2[0], p2[1])

            if not check_label_collision(cand_rect, drawn_rects):
                best_p1, best_p2, best_y = p1, p2, cand_y
                drawn_rects.append(cand_rect)
                break

        if best_p1 is None:  # Fallback if crowded
            best_p1 = (px - int(tw/2) - 6, tag_y - th - 4)
            best_p2 = (px + int(tw/2) + 6, tag_y + 4)
            best_y = tag_y

        cv2.rectangle(img, best_p1, best_p2, (245, 245, 245), -1)
        cv2.rectangle(img, best_p1, best_p2, (40, 40, 40), 1)
        cv2.putText(img, name_tag, (px - int(tw/2), best_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)

def run_volumetric_overlay():
    image_path = "broadcast_frame.png"
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"❌ Error: Could not find '{image_path}'!")
        return

    print("🚀 Running YOLOv8 Detection with Smart Kit Matching...")
    model = YOLO('yolov8s.pt')
    results = model(frame)[0]

    arsenal_players, psg_players = [], []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        if cls_id == 0 and confidence > 0.18:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            feet_x, feet_y = int((x1 + x2) / 2), int(y2)
            box_h = y2 - y1

            # Sample chest kit color
            chest = frame[y1 + int(box_h*0.15):y1 + int(box_h*0.45), x1 + int((x2-x1)*0.2):x2 - int((x2-x1)*0.2)]
            if chest.size == 0: continue
            
            b_avg, g_avg, r_avg = np.mean(chest[:, :, 0]), np.mean(chest[:, :, 1]), np.mean(chest[:, :, 2])
            
            # Filter referee (Teal / High Green + Blue mix)
            if g_avg > 140 and b_avg > 120 and r_avg < 100:
                continue # Skip referee

            # Classify Arsenal (Red) vs PSG (Navy/Dark)
            if r_avg > (b_avg * 1.18) and r_avg > 65:
                arsenal_players.append((feet_x, feet_y, box_h))
            else:
                psg_players.append((feet_x, feet_y, box_h))

    # Sort each team from left to right on the pitch
    arsenal_players = sorted(arsenal_players, key=lambda p: p[0])
    psg_players = sorted(psg_players, key=lambda p: p[0])

    arsenal_lineup = ["RAYA", "GABRIEL", "SALIBA", "MOSQUERA", "HINCAPIE", "RICE", "LEWIS-SKELLY", "ODEGAARD", "TROSSARD", "SAKA", "HAVERTZ"]
    psg_lineup = ["SAFONOV", "HAKIMI", "MARQUINHOS", "PACHO", "MENDES", "VITINHA", "JOAO NEVES", "RUIZ", "DOUE", "DEMBELE", "KVARATSKHELIA"]

    drawn_rects = []

    # Render Arsenal Players
    for idx, (fx, fy, bh) in enumerate(arsenal_players):
        name = arsenal_lineup[idx] if idx < len(arsenal_lineup) else f"ARS #{idx+1}"
        draw_3d_volumetric_spotlight(frame, (fx, fy), bh, drawn_rects, color=(220, 255, 220), name_tag=name)

    # Render PSG Players
    for idx, (fx, fy, bh) in enumerate(psg_players):
        name = psg_lineup[idx] if idx < len(psg_lineup) else f"PSG #{idx+1}"
        draw_3d_volumetric_spotlight(frame, (fx, fy), bh, drawn_rects, color=(200, 220, 255), name_tag=name)

    cv2.imwrite("volumetric_spotlight_result.png", frame)
    print("🎉 Success! Corrected overlay saved to volumetric_spotlight_result.png")

if __name__ == "__main__":
    run_volumetric_overlay()