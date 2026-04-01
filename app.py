import os
import cv2
import base64
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
from deepface import DeepFace
import pytesseract
def normalize_image_for_verification(image_path):
    """Applies CLAHE to improve contrast in the ID card photo."""
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return

        # Convert the image to grayscale, as CLAHE works on single channels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create a CLAHE object (we can tune the parameters if needed)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # Convert the enhanced grayscale image back to a 3-channel BGR image
        # because DeepFace expects a color image.
        enhanced_color = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # Overwrite the temporary file with the new, enhanced image
        cv2.imwrite(image_path, enhanced_color)
        print(f"Image at {image_path} has been normalized.")
    except Exception as e:
        print(f"Could not normalize image at {image_path}. Error: {e}")

# --- IMPORTANT: Configure Tesseract Path if needed ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
except Exception:
    print("Tesseract not found at default location. Please update the path in app.py if you have it installed elsewhere.")

app = Flask(__name__)
# Use eventlet for WebSocket stability
socketio = SocketIO(app, async_mode='eventlet')

# --- Configuration ---
REGISTRATIONS_DIR = 'registrations'
os.makedirs(REGISTRATIONS_DIR, exist_ok=True)
mp_face_detection = mp.solutions.face_detection
user_states = {}

# --- Page Routes ---
@app.route('/')
def register_page():
    """Serves the user registration page with liveness check."""
    return render_template('register.html')

@app.route('/verify-page')
def verify_page():
    """Serves the ID card verification page."""
    return render_template('verify.html')

# --- WebSocket Handlers for Liveness Registration ---
@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)
    user_states[request.sid] = {
        "stage": 0,
        "stages": ["Look Straight", "Turn Head Left", "Turn Head Right", "Look Straight Again"],
        "captured_frame": None
    }
    emit('server_message', {'message': 'Please Look Straight to Begin'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)
    if request.sid in user_states:
        del user_states[request.sid]

@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid
    if sid not in user_states or user_states[sid]["stage"] >= len(user_states[sid]["stages"]):
        return

    state = user_states[sid]
    current_stage_name = state["stages"][state["stage"]]
    
    image_data = base64.b64decode(data['image_data'].split(',')[1])
    frame = np.array(Image.open(BytesIO(image_data)))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            detection = results.detections[0]
            nose_tip = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
            nose_x = nose_tip.x

            if current_stage_name == "Look Straight":
                if 0.4 < nose_x < 0.6:
                    advance_stage(sid)
                    emit_next_instruction(sid)
            
            # --- CORRECTED LOGIC FOR MIRRORED VIEW ---
            elif current_stage_name == "Turn Head Left":
                # For a mirrored view, turning left moves the nose to the left of the screen (lower x value)
                if nose_x < 0.35:
                    advance_stage(sid)
                    emit_next_instruction(sid)
            
            # --- CORRECTED LOGIC FOR MIRRORED VIEW ---
            elif current_stage_name == "Turn Head Right":
                # For a mirrored view, turning right moves the nose to the right of the screen (higher x value)
                if nose_x > 0.65:
                    advance_stage(sid)
                    emit_next_instruction(sid)

            elif current_stage_name == "Look Straight Again":
                if 0.4 < nose_x < 0.6:
                    state["captured_frame"] = frame 
                    if is_image_clear(frame, threshold=40.0):
                        if save_registration_photo(frame, data['user_id']):
                            emit('liveness_success', {'message': 'Liveness Confirmed! Photo Saved.'})
                            advance_stage(sid)
                        else:
                            emit('registration_failed', {'message': 'Error saving photo. Please try again.'})
                            reset_user_state(sid)
                    else:
                        emit('registration_failed', {'message': 'Photo is blurry. Please try again.'})
                        reset_user_state(sid)
        else:
            emit('server_message', {'message': 'No face detected.'})

def is_image_clear(frame, threshold=40.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Image clarity score: {laplacian_var:.2f}")
    return laplacian_var > threshold

def advance_stage(sid):
    if sid in user_states:
        user_states[sid]["stage"] += 1

def reset_user_state(sid):
    if sid in user_states:
        user_states[sid]["stage"] = 0
        user_states[sid]["captured_frame"] = None
        emit('server_message', {'message': 'Restarting... Please Look Straight'})

def emit_next_instruction(sid):
    state = user_states[sid]
    if state["stage"] < len(state["stages"]):
        emit('server_message', {'message': f'Please {state["stages"][state["stage"]]}'})

def save_registration_photo(frame, user_id):
    if frame is not None and user_id and user_id.strip():
        filename = "".join(c for c in user_id if c.isalnum() or c in ('-', '_')).rstrip()
        filename = f"{filename}.jpg"
        filepath = os.path.join(REGISTRATIONS_DIR, filename)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        print(f"Attempting to save photo for user_id: '{user_id}' at path: {filepath}")
        
        try:
            success = cv2.imwrite(filepath, frame_bgr)
            if success:
                print(f"SUCCESS: Saved clear registration photo to {filepath}")
                return True
            else:
                print(f"ERROR: cv2.imwrite() failed to save image to {filepath}. Check folder permissions.")
                return False
        except Exception as e:
            print(f"ERROR: An exception occurred while saving the image: {e}")
            return False
    else:
        print(f"ERROR: Cannot save photo. Frame is valid: {frame is not None}. User ID is valid: {user_id and user_id.strip()}")
        return False

# --- ID Card Verification Endpoint (Face Only) ---
@app.route('/verify-with-id', methods=['POST'])
def verify_with_id():
    user_id = request.form.get('user_id')
    live_image_file = request.files.get('live_image')
    id_card_file = request.files.get('id_card_image')

    if not all([user_id, live_image_file, id_card_file]):
        return jsonify({"verified": False, "reason": "Missing required data."})

    try:
        # DeepFace works with image file paths directly. Let's save the uploaded files temporarily.
        live_image_path = os.path.join(REGISTRATIONS_DIR, f"temp_live_{user_id}.jpg")
        id_card_path = os.path.join(REGISTRATIONS_DIR, f"temp_id_{user_id}.jpg")
        live_image_file.save(live_image_path)
        id_card_file.save(id_card_path)

        # Normalize the ID card photo before verification
        normalize_image_for_verification(id_card_path)
        normalize_image_for_verification(live_image_path)


        # Step 1: Face Match (Live Face vs. ID Card) using FaceNet
        # The 'verify' function returns a dictionary. We check the 'verified' key.
        result_id = DeepFace.verify(
            img1_path=live_image_path,
            img2_path=id_card_path,
            model_name='Facenet512'
        )
        if not result_id['verified']:
            return jsonify({"verified": False, "reason": "Live face does not match ID card photo."})

        # Step 2: Final Security Check (Live Face vs. Original Registration Photo)
        registered_image_path = os.path.join(REGISTRATIONS_DIR, f"{user_id}.jpg")
        if not os.path.exists(registered_image_path):
             return jsonify({"verified": False, "reason": "User has not completed the liveness registration."})

        result_reg = DeepFace.verify(
            img1_path=live_image_path,
            img2_path=registered_image_path,
            model_name='Facenet'
        )
        if not result_reg['verified']:
            return jsonify({"verified": False, "reason": "Face does not match original registration photo."})

        # Clean up temporary files
        os.remove(live_image_path)
        os.remove(id_card_path)

        return jsonify({"verified": True, "reason": "Face Verified Successfully!"})

    except ValueError as e:
        # This error is often thrown by DeepFace if no face is found
        print(f"DeepFace error: {e}")
        return jsonify({"verified": False, "reason": f"Could not find a face in one of the images. Details: {e}"})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"verified": False, "reason": "An server error occurred."})


if __name__ == '__main__':
    socketio.run(app, debug=True)
