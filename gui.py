from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from keras.models import load_model
import numpy as np
import random

root = Tk()
root.title('Face Recognition')
root.geometry('1350x720')

style = Style()
style.configure('TButton', font =('calibri', 15, 'bold'))

model_gender = load_model('D:/TERM 6/AI/Cuoi_ky/model/gender_detection_final.h5')
model_age = load_model('D:/TERM 6/AI/Cuoi_ky/model/age_detection_final.h5')

label = Label(root, text='Predict Age and Gender', font=('arial', 20, 'bold'))
label.grid(row=0, column=1, columnspan=2,padx=(70,0), pady=(40,0), sticky='w')

canv = Canvas(root, width=500, height=450, bg='white')
label1 = Label(canv, background='white')
canv.grid(row=1, column=0, padx=(100,0), pady=(50,0))
canv.create_rectangle(1,1,503,453, outline= 'black', width=4, fill='white')


def select_file():
	global imgToInsert, yourImage, stop, c, canv
	stop = 1
	c=1
	yourImage=filedialog.askopenfilename(title = "Select your image", filetypes = [("Image Files","*.png"),("Image Files","*.jpg")])
	imgFile=Image.open(yourImage)
	imgfile = imgFile.resize((500, 450))
	imgToInsert=ImageTk.PhotoImage(imgfile)
	canv.create_image(1.5,1.5, anchor=NW, image=imgToInsert)

	x = y = 0
	global start_x ,start_y
	def on_button_press(event):
		# save mouse drag start position
		global start_x, start_y, rect
		start_x = event.x
		start_y = event.y

		# create rectangle if not yet exist
		#if notrect:
		rect =canv.create_rectangle(x,y, 1, 1, width=3, outline='red')

	def on_move_press(event):
		global start_x, start_y, rect, curX, curY
		curX, curY = (event.x, event.y)

		# expand rectangle as you drag the mouse
		canv.coords(rect,start_x,start_y, curX, curY)

	def on_button_release(event):
		pass

	canv.bind("<ButtonPress-1>", on_button_press)
	canv.bind("<B1-Motion>", on_move_press)
	canv.bind("<ButtonRelease-1>", on_button_release)
		

names_man = ['John', 'Bill', 'Luck', 'James', 'Robert', 'Michael', 'David', 'Thomas', 'Matthew', 'Anthony','Paul', 'Kevin','Thanh','Nhan','Tuan']
names_woman = ['Mary','Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Jessica', 'Nancy','Emily', 'Amanda', 'Laura', 'Emma', 'Christine', 'Helen', 'Catherine', 'Catherine', 'Lauren']
def predict():
	global yourImage, rec, c, start_x, start_y, curX, curY
	classes = ['man','woman']
	ages = ['1-2', '3-9', '10-18', '19-29', '30-45', '46-65', '66-85', '86-100']
	print(c)
	if c == 1:
		# print(start_x, start_y, curX, curY)
		res = []
		img = cv2.imread(yourImage)
		img = cv2.resize(img, (500, 450))
		image = img[start_y:curY, start_x:curX]
		cv2.imshow('img', image)

		fa = cv2.resize(image, (224, 224))
		face = fa.astype('float32')/255.0
		face = np.expand_dims(np.array(face), axis=0)

		fa_age = cv2.resize(image, (150, 150))
		face_age = fa_age.astype('float32')/255.0
		face_age =np.expand_dims(np.array(face_age), axis=0)

		conf = model_gender.predict(face)[0]
		# print(conf)
		# get label with max accuracy
		idx = np.argmax(conf)
		gen = classes[idx]
		
		age = model_age.predict(face_age)[0]
		age = ages[np.argmax(age)]

		if gen == 'man':
			name = random.choice(names_man)
		else:
			name = random.choice(names_woman)
		res.append((f'{name}', f'{gen}', f'{age}'))

		# add data to the treeview
		for face in res:#
			tree.insert('', END, values=face)
		
	elif c == 0:
		res = []
		fa = cv2.resize(rec, (224, 224))
		face = fa.astype('float32')/255.0
		face = np.expand_dims(np.array(face), axis=0)
		
		fa_age = cv2.resize(rec, (150, 150))
		face_age = fa_age.astype('float32')/255.0
		face_age =np.expand_dims(np.array(face_age), axis=0)

		conf = model_gender.predict(face)[0] 
		# get label with max accuracy
		idx = np.argmax(conf)
		gen = classes[idx]
		
		age = model_age.predict(face_age)[0]
		age = ages[np.argmax(age)]

		if gen == 'man':
			name = random.choice(names_man)
		else:
			name = random.choice(names_woman)
		res.append((f'{name}', f'{gen}', f'{age}'))

		# add data to the treeview
		for face in res:#
			tree.insert('', END, values=face)


def clear():
	selected_item = tree.selection()[0]
	tree.delete(selected_item)

vid = cv2.VideoCapture(0)
global stop
stop = 0
c = 1

def open_camera():
	# Capture the video frame by frame
	global label1, stop, c, cv2image

	if stop == 1 :
		label1.destroy()
		label1 = Label(canv, background='white')
		stop = 0
		c = 1

	elif stop == 0:
		c = 0
		ret, frame = vid.read()
		label1.place(x=0.05, y=0.05)

		# Convert image from one color space to other
		cv2image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		cv2.rectangle(cv2image, (200,100), (420,350), (0,255,0), 2)
		img = Image.fromarray(cv2image)
		img = img.resize((500, 450))
		
		imgtk = ImageTk.PhotoImage(image = img)
		label1.imgtk = imgtk
		
		label1.configure(image=imgtk)
		label1.after(20, open_camera)

def take_pic():
	global cv2image, c, rec, label1
	rec = cv2.cvtColor(cv2image[100:350, 200:420], cv2.COLOR_RGB2BGR)
	cv2.imshow('img', rec)
	label1.destroy()
	label1 = Label(canv, background='white')
	label1.place(x=0.05, y=0.05)
	rec1 = cv2.cvtColor(rec, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(rec1)
	img = img.resize((500, 450))
	imgtk = ImageTk.PhotoImage(image = img)
	label1.imgtk = imgtk
	label1.configure(image=imgtk)
	
    
btn1 = Button(root, text = 'Insert image', width=18, style='TButton', command =lambda: select_file())
btn1.grid(row=2, column=0,padx=(100, 0), pady=(20,0), sticky='w')

btn2 = Button(root, text = 'Predict',width=17, style='TButton', command =lambda: predict())
btn2.grid(row=2, column=0, padx=(100,0), pady=(20,0), sticky='e')

btn3 = Button(root, text = 'Delete', style='TButton', command = clear)
btn3.grid(row=2, column=1, padx=(335,0), sticky='w')

btn4 = Button(root, text = 'Open camera', width=18, style='TButton', command = open_camera)
btn4.grid(row=3, column=0, padx=(100,0), pady=(20,0), sticky='w')

btn5 = Button(root, text = 'Take picture',width=17, style='TButton', command =lambda: take_pic())
btn5.grid(row=3, column=0, padx=(330,0), pady=(20,0), sticky='e')

columns = ('Name', 'Gender', 'Age')
tree = Treeview(root, columns=columns, height=19, show='headings')
# define headings
tree.column('Name', anchor=CENTER)
tree.heading('Name', text='Name')
tree.column('Gender', anchor=CENTER)
tree.heading('Gender', text='Gender')
tree.column('Age', anchor=CENTER)
tree.heading('Age', text='Age')

tree.grid(row=1, column=1,padx=(100,0), pady=(50, 0), sticky='n')

# add a scrollbar
scrollbar = Scrollbar(root, orient=VERTICAL, command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.grid(row=1, column=2, pady=(47, 0), sticky='ns')

root.mainloop()
