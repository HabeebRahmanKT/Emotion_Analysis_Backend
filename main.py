from fer import FER
from fer import Video
from flask import Flask
import cv2
import json
import os
import matplotlib as plt
import time

app = Flask(__name__)


def record_video():
    print("Info: Video recording started")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    print("Info: Video recording started")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    for frame_count in range(0, 40):
        ret, frame = cap.read()

        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    print("Info: Video recording ended")
    cap.release()
    out.release()

    cv2.destroyAllWindows()


@app.route("/")
def get_emotion_result():
    record_video()
    video_file = []
    for file in os.listdir():
        if file.endswith(".mp4") and file not in video_file:
            emotion_detector = FER(mtcnn=True)

            video_file.append(file)
            path_to_video = file
            video = Video(path_to_video)

            print("Info: Emotion Analyzing started")
            result = video.analyze(emotion_detector, save_video=False, frequency=8, save_frames=False,
                                   annotate_frames=False)

            print("Info: Emotion Analyzing ended")

            emotion_result = {'angry0': 0.0, 'happy0': 0.0, 'sad0': 0.0, 'neutral0': 0.0, 'fear0': 0.0}
            for frames in result:
                for emotion in emotion_result:
                    emotion_result[emotion] = (emotion_result[emotion] + frames[emotion])

            for emotion in emotion_result:
                emotion_result[emotion] = emotion_result[emotion] / len(result)

            df = video.to_pandas(result)
            print(df)

            return_data = {
                "Angry": emotion_result['angry0'],
                "Happy": emotion_result['happy0'],
                "Sad": emotion_result['sad0'],
                "Fear": emotion_result['fear0'],
                "Neutral": emotion_result['neutral0']
            }

            return json.dumps(return_data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

    # while True:
    #     # record_video()
    #     time.sleep(30)
