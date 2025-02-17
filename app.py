import gradio as gr
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
#import torch
#import xml.etree.ElementTree as ET
#from glob import glob

# Load YOLO model
model = YOLO("ANPR_ver15.pt")  # Load your trained model

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# def anpr_pipeline(image):
#     # Convert Gradio image to OpenCV format
#     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     # License plate detection using YOLO
#     results = model(image)

#     plates = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cropped_plate = image[y1:y2, x1:x2]

#             # OCR for text recognition
#             ocr_result = ocr.ocr(cropped_plate, cls=True)
#             plate_text = " ".join([entry[1][0] for entry in ocr_result[0]]) if ocr_result[0] else "N/A"
#             print("plate_text",plate_text)
#             plates.append((plate_text))

#     return plates


# Define a simple confusion mapping for common OCR errors
digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '9': 'P'}
letter_to_digit = {v: k for k, v in digit_to_letter.items()}


def extract_license_plate(ocr_result, confidence_threshold=0.5, size_threshold_factor=0.5):
    """
    Extracts and cleans text from OCR output, then applies a simple HMM correction
    based on the expected license plate format: AAXXAAXXXX (where A=alphabet, X=digit).
    """
    # If OCR result is empty or improperly formatted, return a default value
    if not ocr_result or not ocr_result[0]:
        return "No license plate detected"

    # Initialize a list to hold texts that pass filtering
    detected_texts = []
    max_area = 0  # To track the largest detected text area

    # Loop through OCR detections
    for detection in ocr_result[0]:  # Assuming detections are in the first element
        if len(detection) > 0:
            text = detection[1][0]  # Extract the text
            confidence = detection[1][1]  # Extract confidence score
            box = detection[0]  # Bounding box (coordinates)

            # Calculate the area of the bounding box (width * height)
            width = abs(box[1][0] - box[0][0])
            height = abs(box[2][1] - box[1][1])
            area = width * height

            # Update maximum area for size comparison
            max_area = max(max_area, area)

            # Filter based on confidence score and size (compared to largest detected text area)
            if confidence > confidence_threshold and area > (max_area * size_threshold_factor):
                detected_texts.append(text)

    # Concatenate all filtered texts into one string
    final_text = " ".join(detected_texts)

    # Remove non-alphanumeric characters using regex
    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', final_text)

    # --- HMM Correction Step ---
    # We expect a license plate of format: AAXXAAXXXX
    # (positions 0-1 and 4-5 are letters; positions 2-3 and 6-9 are digits)

    def hmm_correct_plate(obs_text):
        expected_types = ['letter', 'letter', 'digit', 'digit',
                          'letter', 'letter', 'digit', 'digit', 'digit', 'digit']
        n = len(expected_types)

        # If the observed text is not 10 characters, you might want to handle it differently.
        # For now, we only correct if we have exactly 10 characters.
        if len(obs_text) != n:
            return obs_text  # Or return an error/default value

        # Candidate sets for each expected type
        letter_candidates = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        digit_candidates = list("0123456789")

        # Define emission probability function.
        # We assume:
        #   - if observed char is of the expected type and matches candidate: 0.9
        #   - if observed char is of the expected type but not equal: small probability
        #   - if observed char is of the opposite type but can be mapped via confusion: moderate probability
        #   - otherwise, very low probability.
        def emission_prob(expected_type, candidate, observed):
            candidate = candidate.upper() if expected_type == 'letter' else candidate
            observed = observed.upper() if expected_type == 'letter' else observed
            if expected_type == 'letter':
                if observed.isalpha():
                    return 0.9 if candidate == observed else 0.1 / (len(letter_candidates) - 1)
                elif observed.isdigit():
                    # Allow mapping from digit to letter if common OCR confusion exists
                    if digit_to_letter.get(observed, None) == candidate:
                        return 0.5
                    else:
                        return 0.01
                else:
                    return 0.01
            else:  # expected digit
                if observed.isdigit():
                    return 0.9 if candidate == observed else 0.1 / (len(digit_candidates) - 1)
                elif observed.isalpha():
                    if letter_to_digit.get(observed, None) == candidate:
                        return 0.5
                    else:
                        return 0.01
                else:
                    return 0.01

        # For simplicity, we use uniform (or identity) transition probabilities.
        # Our state sequence is fixed by the format so each position is independent given the observation.
        # We still demonstrate a Viterbi algorithm.

        dp = []  # dp[i] will be a dict mapping candidate char at position i to probability
        backpointer = []  # To recover the best sequence

        # Initialization for position 0
        current_candidates = letter_candidates if expected_types[0] == 'letter' else digit_candidates
        dp0 = {}
        bp0 = {}
        for c in current_candidates:
            dp0[c] = emission_prob(expected_types[0], c, obs_text[0])
            bp0[c] = None
        dp.append(dp0)
        backpointer.append(bp0)

        # Recursion for positions 1 to n-1
        for i in range(1, n):
            current_candidates = letter_candidates if expected_types[i] == 'letter' else digit_candidates
            dp_curr = {}
            bp_curr = {}
            for curr in current_candidates:
                max_prob = -1
                best_prev = None
                # Since transition probabilities are uniform, we only multiply by emission probability.
                for prev, prev_prob in dp[i - 1].items():
                    prob = prev_prob * emission_prob(expected_types[i], curr, obs_text[i])
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev
                dp_curr[curr] = max_prob
                bp_curr[curr] = best_prev
            dp.append(dp_curr)
            backpointer.append(bp_curr)

        # Termination: pick the candidate at the last position with maximum probability.
        last_candidates = dp[-1]
        best_last = max(last_candidates, key=last_candidates.get)

        # Backtrace to retrieve the best candidate sequence.
        best_sequence = [best_last]
        for i in range(n - 1, 0, -1):
            best_sequence.insert(0, backpointer[i][best_sequence[0]])

        return "".join(best_sequence)

    # Apply HMM correction if the cleaned text is 10 characters long;
    # otherwise, return the cleaned text as is.
    if len(cleaned_text) == 10:
        print("Calling hmm correct plate")
        corrected_text = hmm_correct_plate(cleaned_text)
    else:
        print("directly assigned cleaned text")
        corrected_text = cleaned_text

    return corrected_text


def anpr_pipeline1(image):
    # Convert Gradio image to OpenCV format
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # License plate detection using YOLO
    results = model.predict(image, imgsz=640)

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        # Ensure bounding box is within image boundaries
        h, w, _ = image.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        # Crop the license plate
        cropped_plate = image[y1:y2, x1:x2]

        # logging.getLogger("ppocr").setLevel(logging.ERROR)  # Suppress PaddleOCR debug logs
        ocr = PaddleOCR(use_angle_cls=True, lang="en")

        # Run OCR
        ocr_result = ocr.ocr(cropped_plate, cls=True)

        # Extract license plate text
        license_plate_text = extract_license_plate(ocr_result)

        if license_plate_text:
            # Draw bounding box around detected plate
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Overlay detected text on the bounding box
            # Black outline (shadow)
            cv2.putText(image, license_plate_text, (x1 -13, y1 - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)

            # White main text
            cv2.putText(image, license_plate_text, (x1-14, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        #    cv2.putText(image, license_plate_text, (x1, y1 - 10),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            #return image
            return image, license_plate_text
        else:
            print("License Plate Not Detected", license_plate_text)
            return 0

# Gradio UI
iface = gr.Interface(
    fn=anpr_pipeline1,
    inputs=gr.Image(type="numpy"),
    outputs=["image", "text"],
    title="Automatic Number Plate Recognition (ANPR)",
    description="Upload an image with a vehicle, and the model will detect and extract the license plate number."
)

iface.launch()
