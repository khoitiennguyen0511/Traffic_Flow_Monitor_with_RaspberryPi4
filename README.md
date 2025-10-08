**Link Demo:**

https://www.youtube.com/watch?v=cf1kmcOJJgg

**Steps to run code**

**Step 1: Install virtual environment**

```
python -m venv pedestrian_env
```

- On Window:

```
pedestrian_env\Scripts\activate
```

- On MacOs/Linux:

```
source pedestrian_env/bin/activate
```

Once enabled, you will see a virtual environment at the beginning of the command line, for example:

```
(pedestrian_env) PS C:\Users\tienk>
```

**Step 2: Clone the repository and install the dependencies**

```
git clone https://github.com/khoitiennguyen0511/Pedestrian_Tracking_and_Counting.git

```

- Goto the cloned folder.

```
cd Pedestrian_Tracking_and_Counting
```

```
pip install -r requirements.txt
```

```
winget install ffmpeg
```

**Step 3: Run code**

```
streamlit run app.py
```

Then, a window appears and you choose **video.mp4** file in my cloned repository. If you want to choose another video, you need to choose a video with the format (.mp4, .avi, .webm, .wpeg4 and limit 200MB)
