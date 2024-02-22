import cv2
import mediapipe as mp
import pyautogui
import math
import threading
import time
pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
click=[False, 0, 0, ()]
click_threshold = 0.18
sigma_threshold = 180
scroll_tresh = 15
mouse = [False, 0, 0, ()]
cap = cv2.VideoCapture(0)
drawing = []
def get_x_y(landmark, image):
  height, width, _ = image.shape
  return [
    int(landmark.x * width),
    int(landmark.y * height)
  ]

def write_password(password):
  for i in password:
    try:
      int(i)
      pyautogui.hotkey("shift", i)
    except:
      pyautogui.typewrite(str(i))
    time.sleep(0.05)
  pyautogui.press("enter")

def get_distance(a, b):
  return int(
    (
      (b[0] - a[0]) ** 2
      + (b[1] - a[1]) **2
    ) ** 0.5
  )

def compute_landmarks(image, hand_landmarks):
  global click, mouse
  hands_found = len(hand_landmarks)
  if hands_found == 1:
    hand_landmarks = hand_landmarks[0]
    index_coordinates = get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], image)
    middle_coordinates = get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], image)
    thumb_coordinates = get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP], image)
    ring_coordinates = get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP], image)
    pinky_coordinates = get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP], image)
    wrist_coordinates = get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST], image)
    """pointer_coordinates_g = (
      int( ( (2*int( (index_coordinates[0] + middle_coordinates[0]) / 2 )) + thumb_coordinates[0] ) / 3 ),
      int( ( (2*int( (index_coordinates[1] + middle_coordinates[1]) / 2 )) + thumb_coordinates[1] ) / 3 ),
    )
    pointer = pointer_coordinates_g"""
    pointer = (
      int( ( (int( (index_coordinates[0] + thumb_coordinates[0]) / 2 ))  )),
      int( ( (int( (index_coordinates[1] + thumb_coordinates[1]) / 2 ))  ))
    )
    distance_i_m = get_distance(index_coordinates, middle_coordinates)
    distance_i_t = get_distance(index_coordinates, thumb_coordinates)
    distance_m_t = get_distance(thumb_coordinates, middle_coordinates)
    reference_distance = get_distance(get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], image), get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP], image))
    mouse_distance = get_distance(get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP], image), get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP], image))
    if reference_distance != 0 and int(mouse_distance/(reference_distance*1.7)) == 0:
      """if mouse[0]:
        mouse_move = (
          (int((mouse[3][0] - thumb_coordinates[0])*10/reference_distance)*10),
          -(int((mouse[3][1] - thumb_coordinates[1])*10/reference_distance)*10)
        )
        threading.Thread(target=lambda: pyautogui.moveRel(mouse_move[0], mouse_move[1])).start()
      mouse = [True, thumb_coordinates]"""
      if mouse[0]:
        #thumb and index in contact
        if (time.time() - mouse[1]) > click_threshold:
          mouse_move = (
            (int((mouse[3][0] - index_coordinates[0])*10/reference_distance)*10),
            -(int((mouse[3][1] - index_coordinates[1])*10/reference_distance)*10)
          )
          mouse[3] = index_coordinates
          threading.Thread(target=lambda: pyautogui.moveRel(mouse_move[0], mouse_move[1])).start()
          cv2.rectangle(image, (mouse[3][0]-scroll_tresh, mouse[3][1]-scroll_tresh), (mouse[3][0]+scroll_tresh, mouse[3][1]+scroll_tresh), (25, 25, 25), -1)
      else:
        print("mouse")
        mouse = [True, time.time(), 0, index_coordinates]
    else:
      if mouse[0]:
        if ((time.time() - mouse[1]) <= click_threshold):
          print("RIGHT!")
          threading.Thread(target=pyautogui.click).start()
      mouse=[False, 0, 0, mouse[3]]
    if reference_distance != 0 and int(distance_i_t/(reference_distance*1.9)) == 0 and not mouse[0]:
      drawing.append(pointer)
      if click[0]:
        #thumb and index in contact
        if (time.time() - click[1]) > click_threshold:
          if not click[2]:
            click[2] = 1
            click[3] = pointer
          if abs(pointer[1]-click[3][1]) >= scroll_tresh or abs(pointer[0]-click[3][0]) >= scroll_tresh:
            print("Scrolling : ", end="")
            if abs(pointer[1]-click[3][1]) >= abs(pointer[0]-click[3][0]):
              if pointer[1] <= click[3][1]-scroll_tresh:
                print("UP")
                threading.Thread(target=lambda: pyautogui.scroll(-1)).start()
              elif pointer[1] >= click[3][1]+scroll_tresh:
                print("DOWN")
                threading.Thread(target=lambda: pyautogui.scroll(1)).start()
              pass
            else:
              if pointer[0] <= click[3][0]-scroll_tresh:
                print("RIGHT")
                threading.Thread(target=lambda: pyautogui.hscroll(-1)).start()
              elif pointer[0] >= click[3][0]+scroll_tresh:
                print("LEFT")
                threading.Thread(target=lambda: pyautogui.hscroll(1)).start()
              pass
          cv2.rectangle(image, (click[3][0]-scroll_tresh, click[3][1]-scroll_tresh), (click[3][0]+scroll_tresh, click[3][1]+scroll_tresh), (25, 25, 25), -1)
          cv2.rectangle(image, (click[3][0]-scroll_tresh, 0), (click[3][0]+scroll_tresh, 720), (25, 25, 25), 2)
          cv2.rectangle(image, (0, click[3][1]-scroll_tresh), (1080, click[3][1]+scroll_tresh), (25, 25, 25), 2)
      else:
        print("Clicking")
        click = [True, time.time(), 0, ()]
    else:
      if click[0]:
        if ((time.time() - click[1]) <= click_threshold):
          print("LEFT!")
          threading.Thread(target=lambda: pyautogui.rightClick()).start()
      click=[False, 0, 0, ()]
    cv2.circle(image, index_coordinates, 5, (0, 0, 255), -1)
    cv2.circle(image, middle_coordinates, 5, (0, 0, 255), -1)
    cv2.circle(image, thumb_coordinates, 5, (0, 0, 255), -1)
    cv2.circle(image, ring_coordinates, 5, (0, 0, 255), -1)
    cv2.circle(image, pinky_coordinates, 5, (0, 0, 255), -1)
    cv2.circle(image, wrist_coordinates, 5, (0, 0, 255), -1)
    cv2.circle(image, get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], image), 5, (255, 0, 0), -1)
    cv2.circle(image, get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP], image), 5, (255, 0, 0), -1)
    cv2.circle(image, get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP], image), 5, (255, 0, 0), -1)
    cv2.circle(image, get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP], image), 5, (255, 0, 0), -1)
    cv2.circle(image, get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], image), 5, (255, 0, 0), -1)
    cv2.circle(image, get_x_y(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC], image), 5, (255, 0, 0), -1)
    cv2.circle(image, pointer, (10 if click[0] else 5), (((255, 255, 0) if click[2] else (0, 255, 0)) if click[0] else (0, 0, 0)), -1)
  elif hands_found == 2:
    print("two hands found")
  else:
    print("a loooot of hands found")
    for (hand_number, hand_landmarks) in enumerate(results.multi_hand_landmarks):
      pass

# Set the capture resolution to the native resolution
with mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image=cv2.resize(image, (1080, 720))
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    cv2.rectangle(image, (0, 0), (1080, 720), (255, 255, 255), -1)
    rectangle=((200, 40), (40, 200))
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      compute_landmarks(image, results.multi_hand_landmarks)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()