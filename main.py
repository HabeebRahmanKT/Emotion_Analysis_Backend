from fer import FER
from fer import Video
from flask import Flask
import json
import os

app = Flask(__name__)


@app.route("/")
def get_emotion_result():
    video_file = []
    for file in os.listdir():
        if file.endswith(".mp4") and file not in video_file:
            emotion_detector = FER(mtcnn=True)

            video_file.append(file)
            path_to_video = file
            video = Video(path_to_video)

            result = video.analyze(emotion_detector, save_frames=False, save_video=False, frequency=8)

            # Create Pandas DataFrame with emotion data
            emotions_df = video.to_pandas(result)
            print(emotions_df)

            emotion_result = {'angry0': 0.0, 'happy0': 0.0, 'sad0': 0.0, 'neutral0': 0.0, 'fear0': 0.0}
            for frames in result:
                for emotion in emotion_result:
                    emotion_result[emotion] = (emotion_result[emotion] + frames[emotion])

            for emotion in emotion_result:
                emotion_result[emotion] = emotion_result[emotion] / len(result)

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
    # record_video()
    # get_emotion_result()
