import cv2
import time
import mediapipe as mp

class HandTracker:
    def __init__(self):
        # Initialize MediaPipe hands and drawing utilities
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.pTimer = 0

    def process_frame(self, img):
        # Convert the BGR image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        # Check if any hand is detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    if id == 4:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                # Draw landmarks on the image
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0):
        lmList = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

    def calculate_fps(self):
        cTimer = time.time()
        fps = 1 / (cTimer - self.pTimer)
        self.pTimer = cTimer
        return int(fps)

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = tracker.process_frame(img)

        fps = tracker.calculate_fps()
        cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
