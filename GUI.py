import tkinter as tk
from tkinter import filedialog
from functools import partial
from PIL import Image, ImageTk
from FaceRecognition import face_recognition

def select_file_path(entry):
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def run_face_recognition(entry, image_label_orig, image_label_pred, q_entry, threshold_entry):
    test_image_path = entry.get()
    if test_image_path:
        q = int(q_entry.get())
        threshold = float(threshold_entry.get())
        predicted_label, predicted_image_path = face_recognition(test_image_path, q, threshold)
        print("Predicted Label:", predicted_label)
        display_image(test_image_path, image_label_orig)
        display_image(predicted_image_path, image_label_pred)
    else:
        print("Please select a test image.")

def display_image(image_path, image_label):
    img = Image.open(image_path)
    img.thumbnail((200, 200))  # Resize image if necessary
    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img)
    image_label.image = img  # Keep a reference to the image to prevent garbage collection

# GUI
root = tk.Tk()
root.title("Face Recognition")

# Frame for original and predicted images
frame_images = tk.Frame(root)
frame_images.pack(pady=10)

label_orig = tk.Label(frame_images, text="Original Image")
label_orig.grid(row=0, column=0, padx=10)

label_pred = tk.Label(frame_images, text="Predicted Image")
label_pred.grid(row=0, column=1, padx=10)

image_label_orig = tk.Label(frame_images)
image_label_orig.grid(row=1, column=0)

image_label_pred = tk.Label(frame_images)
image_label_pred.grid(row=1, column=1)

# Frame for q and threshold
frame_params = tk.Frame(root)
frame_params.pack(pady=10)

label_q = tk.Label(frame_params, text="Value of q:")
label_q.grid(row=0, column=0, padx=10)

q_entry = tk.Entry(frame_params)
q_entry.grid(row=0, column=1)

label_threshold = tk.Label(frame_params, text="Threshold:")
label_threshold.grid(row=0, column=2, padx=10)

threshold_entry = tk.Entry(frame_params)
threshold_entry.grid(row=0, column=3)

# Frame for file selection and execution
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

label_select = tk.Label(frame_buttons, text="Select Test Image:")
label_select.grid(row=0, column=0, padx=10)

entry = tk.Entry(frame_buttons, width=50)
entry.grid(row=0, column=1)

button_browse = tk.Button(frame_buttons, text="Browse", command=partial(select_file_path, entry))
button_browse.grid(row=0, column=2, padx=10)

button_run = tk.Button(frame_buttons, text="Run Face Recognition", 
                       command=partial(run_face_recognition, entry, image_label_orig, image_label_pred, q_entry, threshold_entry))
button_run.grid(row=0, column=3, padx=10)

root.mainloop()
