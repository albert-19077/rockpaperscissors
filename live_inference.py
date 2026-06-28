import os
import time
import random
import cv2
import torch
from torchvision import transforms
import config
from model import RockPaperScissorsCNN


def main():
    # Set execution device
    print(f"Executing inference on device: {config.DEVICE}")
    # Resolve saved PyTorch weight files
    weights_path = config.MODEL_PATH
    if not os.path.exists(weights_path) and os.path.exists(f"{weights_path}.pth"):
        weights_path = f"{weights_path}.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Trained weights file '{weights_path}' not found.")
        print("Please train the model first by running: python train.py")
        return
    # Instantiate model and load parameters
    model = RockPaperScissorsCNN(num_classes=config.NUM_CLASSES)
    try:
        state_dict = torch.load(weights_path, map_location=config.DEVICE)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        print("Model weights successfully loaded.")
    except Exception as e:
        print(f"Failed to load as state_dict ({e}). Trying to load full model...")
        try:
            model = torch.load(weights_path, map_location=config.DEVICE)
        except Exception as e_inner:
            print(f"Critical Error: Failed to load weights: {e_inner}")
            return
    model.to(config.DEVICE)
    model.eval()  # Put neural network in evaluation mode
    # Preprocessing pipeline matching dataset_loader.py transformations
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Instantiate webcam feed capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return
    # Set frame resolution to standard 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Scoring and State Variables
    player_score = 0
    ai_score = 0
    game_state = "WAITING"  # Valid game states: WAITING, COUNTDOWN, SHOW_RESULT, NO_MOVE

    countdown_start = 0
    countdown_duration = 3.0  # Seconds

    # Minimum confidence threshold: if model is less than 60% sure, we reject the prediction
    # and ask the player to try again. Increase this value (e.g. 0.70) to make it stricter.
    CONFIDENCE_THRESHOLD = 0.60

    no_move_start = 0  # Timestamp when NO_MOVE state began
    NO_MOVE_DISPLAY_SEC = 2.5  # How many seconds to show the "no move" warning before resetting

    player_choice = ""
    ai_choice = ""
    outcome = ""
    result_color = (255, 255, 255)
    snapshot = None
    # UI Color Definitions
    COLOR_BLUE = (255, 0, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_GRAY = (128, 128, 128)
    COLOR_WHITE = (255, 255, 255)
    COLOR_DARK_BAR = (30, 30, 30)

    def draw_text_with_shadow(img, text, pos, scale, color, thickness=2):
        """Draws double-layered drop-shadow text for visual crispness."""
        cv2.putText(img, text, (pos[0] + 2, pos[1] + 2), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thickness + 1,
                    cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw_viewfinder_corners(img, pt1, pt2, color, thickness=3, length=20):
        """Draws modern corner indicator shapes to align hand placement."""
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(img, pt1, pt2, (100, 100, 100), 1, cv2.LINE_AA)

        # Draw L-shaped corners
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)

    print("Live Inference Game started. Press 'S' to Start, 'Q' to Quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
        # Flip horizontally to match mirror expectations
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        # Define 224x224 bounding box coordinates (centered in frame)
        box_w, box_h = config.IMAGE_SIZE
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        # Extract local guide box crop
        crop = frame[y1:y2, x1:x2]
        if game_state == "WAITING":
            display = frame.copy()
            draw_viewfinder_corners(display, (x1, y1), (x2, y2), COLOR_BLUE)

            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 70), (w, h), COLOR_DARK_BAR, -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            draw_text_with_shadow(display, "Press 'S' to Start Game", (w // 2 - 180, h - 25), 0.8, COLOR_YELLOW)
        elif game_state == "COUNTDOWN":
            display = frame.copy()
            draw_viewfinder_corners(display, (x1, y1), (x2, y2), COLOR_YELLOW)

            elapsed = time.time() - countdown_start
            remaining = countdown_duration - elapsed
            if remaining > 0:
                # Calculate countdown string representation: 3, 2, 1
                num_str = str(int(remaining) + 1)
                t_size = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_DUPLEX, 3.0, 6)[0]
                tx = w // 2 - t_size[0] // 2
                ty = h // 2 + t_size[1] // 2

                # Center text pulsing effect
                pulse = remaining % 1.0
                scale = 3.0 * (0.8 + 0.4 * pulse)
                draw_text_with_shadow(display, num_str, (tx, ty), scale, COLOR_YELLOW, 6)
            else:
                # Countdown finished. Capture snapshot and execute classifier inference.
                snapshot = crop.copy()
                crop_rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)

                # Image processing & evaluation
                input_tensor = preprocess(crop_rgb)
                input_batch = input_tensor.unsqueeze(0).to(config.DEVICE)

                confidence = 0.0
                try:
                    with torch.no_grad():
                        logits = model(input_batch)
                        probs = torch.softmax(logits, dim=1)
                        confidence, idx_tensor = torch.max(probs, dim=1)
                        confidence = confidence.item()
                        idx = idx_tensor.item()
                    player_choice = config.CLASS_NAMES[idx]
                    print(f"Prediction: {player_choice} | Confidence: {confidence:.2%}")
                except Exception as e:
                    print(f"Inference failed ({e}).")
                    confidence = 0.0
                # --- NO MOVE DETECTED CHECK ---
                # If the model's confidence is below our threshold, the hand was not
                # recognized clearly. Show a warning and reset instead of scoring.
                if confidence < CONFIDENCE_THRESHOLD:
                    print(f"Low confidence ({confidence:.2%}). No valid gesture detected.")
                    game_state = "NO_MOVE"
                    no_move_start = time.time()
                else:
                    # Valid gesture detected — proceed with normal game logic
                    ai_choice = random.choice(config.CLASS_NAMES)

                    # Evaluate outcome rules
                    if player_choice == ai_choice:
                        outcome = "Tie!"
                        result_color = COLOR_GRAY
                    elif (player_choice == "rock" and ai_choice == "scissors") or \
                            (player_choice == "paper" and ai_choice == "rock") or \
                            (player_choice == "scissors" and ai_choice == "paper"):
                        outcome = "You Win!"
                        result_color = COLOR_GREEN
                        player_score += 1
                    else:
                        outcome = "You Lose!"
                        result_color = COLOR_RED
                        ai_score += 1
                    game_state = "SHOW_RESULT"
        elif game_state == "NO_MOVE":
            # Show "no gesture detected" warning over the frozen snapshot frame
            display = frame.copy()
            if snapshot is not None:
                display[y1:y2, x1:x2] = snapshot

            # Orange border to indicate a warning (not a win or loss)
            COLOR_ORANGE = (0, 140, 255)
            draw_viewfinder_corners(display, (x1, y1), (x2, y2), COLOR_ORANGE, thickness=4)

            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 140), (w, h), COLOR_DARK_BAR, -1)
            cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

            # Main warning message
            msg = "You didn't make a move!"
            m_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.95, 2)[0]
            draw_text_with_shadow(display, msg, (w // 2 - m_size[0] // 2, h - 90), 0.95, COLOR_ORANGE, 2)
            # Sub-message
            sub = "Lutfen tekrar oynayiniz."
            s_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)[0]
            draw_text_with_shadow(display, sub, (w // 2 - s_size[0] // 2, h - 55), 0.65, COLOR_WHITE, 1)
            # Show countdown to auto-reset
            elapsed_no_move = time.time() - no_move_start
            remaining_reset = max(0.0, NO_MOVE_DISPLAY_SEC - elapsed_no_move)
            draw_text_with_shadow(display, f"Yeniden basliyor: {remaining_reset:.1f}s", (w // 2 - 130, h - 22), 0.55,
                                  COLOR_YELLOW, 1)
            # Auto-reset back to WAITING after NO_MOVE_DISPLAY_SEC seconds
            if elapsed_no_move >= NO_MOVE_DISPLAY_SEC:
                game_state = "WAITING"
                snapshot = None
                print("Auto-reset to WAITING after no move detected.")
        elif game_state == "SHOW_RESULT":
            display = frame.copy()
            # Overlay frozen snapshot of the hand gesture inside guide box area
            if snapshot is not None:
                display[y1:y2, x1:x2] = snapshot

            draw_viewfinder_corners(display, (x1, y1), (x2, y2), result_color, thickness=4)

            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 140), (w, h), COLOR_DARK_BAR, -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            # Draw Player and AI gesture labels
            draw_text_with_shadow(display, f"YOU: {player_choice.upper()}", (30, h - 90), 0.7, COLOR_WHITE)
            draw_text_with_shadow(display, f"AI: {ai_choice.upper()}", (w - 180, h - 90), 0.7, COLOR_WHITE)

            # Center outcome header
            o_size = cv2.getTextSize(outcome, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
            draw_text_with_shadow(display, outcome, (w // 2 - o_size[0] // 2, h - 85), 1.2, result_color, 3)

            # Control instruction footer
            draw_text_with_shadow(display, "Press 'R' to Play Again  |  'Q' to Quit", (w // 2 - 180, h - 25), 0.55,
                                  COLOR_YELLOW, 1)
        # Draw Header Scoreboard
        header = display.copy()
        cv2.rectangle(header, (0, 0), (w, 55), COLOR_DARK_BAR, -1)
        cv2.addWeighted(header, 0.6, display, 0.4, 0, display)

        draw_text_with_shadow(display, f"SCORE - Player: {player_score}   AI: {ai_score}", (20, 35), 0.75, COLOR_WHITE)
        draw_text_with_shadow(display, f"Mode: {game_state}", (w - 150, 35), 0.55, COLOR_YELLOW, 1)
        cv2.imshow("Rock-Paper-Scissors (Modular)", display)
        # Handle keyboard keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if (key == ord('s') or key == ord('S')) and game_state == "WAITING":
            game_state = "COUNTDOWN"
            countdown_start = time.time()
        elif (key == ord('r') or key == ord('R')) and game_state == "SHOW_RESULT":
            game_state = "WAITING"
            snapshot = None
        elif key == ord('q') or key == ord('Q') or key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
