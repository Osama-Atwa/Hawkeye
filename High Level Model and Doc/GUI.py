from tkinter import *
from PIL import Image, ImageTk
import threading
import imageio
# create instance for window
root = Tk()
# set window title
root.title('Video Player')
# read video
video = imageio.get_reader('video2.mp4')
def display_video(label):
   # iterate through video data
   for image in video.iter_data():
      # convert array into image
      img = Image.fromarray(image)
      # Convert image to PhotoImage
      image_frame = ImageTk.PhotoImage(image = img)
      # configure video to the lable
      label.config(image=image_frame)
      label.image = image_frame
# create label for video
video_label = Label(root)
video_label.pack()
# create and start thread
thread = threading.Thread(target=display_video, args=(video_label,))
thread.start()
root.mainloop()