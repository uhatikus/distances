import os
import sys
import read_model
import utils
import time
from tkinter import filedialog
from tkinter import *
from matplotlib import cm
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Fill_window:

	def close_windows(self):
		self.master.destroy()

	def process_clicked(self):
		if len(self.videos) <= 0:
			self.processing.configure(text="ERROR!!! Please, choose videos to analyze")
			return 
		if len(self.target) <= 0:
			self.processing.configure(text="ERROR!!! Please, choose target")
			return 
		self.processing.configure(text="Processing...")
		self.master.update()
		start1 = time.time()
		self.project_dir, self.n_imgs = utils.video_to_frames(self.videos)
		end1 = time.time()
		self.processing.configure(text=self.processing.cget("text")+"\nTime for video_to_frames: " + str(end1 - start1) + " seconds")
		self.master.update()
		start2 = time.time()
		# get model for labeling 
		self.model = utils.get_model(self.project_dir + "images/", self.n_imgs)
		end2 = time.time()
		self.processing.configure(text=self.processing.cget("text")+"\nTime for get_model: " + str(end2 - start2) + " seconds")
		self.master.update()
		start3 = time.time()
		# labeling for cropping
		self.labeled_points_for_cropping = utils.label_points(self.project_dir + "images/", self.model)
		end3 = time.time()
		self.processing.configure(text=self.processing.cget("text")+"\nTime for label_points: " + str(end3 - start3) + " seconds")
		self.master.update()
		start4 = time.time()
		# saved in directory "cropped"
		utils.crop_frames(self.labeled_points_for_cropping, self.project_dir, "cropped", "images")
		end4 = time.time()
		self.processing.configure(text=self.processing.cget("text")+"\nTime for crop_frames: " + str(end4 - start4) + " seconds")
		self.master.update()
		start5 = time.time()
		# get target frames from the cropped using feature extraction 
		utils.get_target_frames(self.project_dir, self.target, "target", "cropped")
		end5 = time.time()
		self.processing.configure(text=self.processing.cget("text")+"\nTime for get_target_frames: " + str(end5 - start5) + " seconds")
		self.master.update()
		start6 = time.time()
		# Structure from Motion
		os.system("colmap automatic_reconstructor --workspace_path " + self.project_dir + " --image_path " + project_dir + "target/")
		end6 = time.time()
		self.processing.configure(text=self.processing.cget("text")+"\nTime for colmap: " + str(end6 - start6) + " seconds")
		self.master.update()
		start7 = time.time()
		# read output of Structure from Motion
		self.cameras, self.images, self.points3D = read_model.read_model(path=self.project_dir+"sparse/0/", ext=".bin")
		end7 = time.time()
		self.processing.configure(text=self.processing.cget("text")+"\nTime for read_model: " + str(end7 - start7) + " seconds")
		self.master.update()
		start8 = time.time()
		# label points of target frames
		self.labeled_points = utils.label_points(self.project_dir + "target/", self.model)
		end8 = time.time()
		self.processing.configure(text=self.processing.cget("text")+"\nTime for label_points: " + str(end8 - start8) + " seconds")
		self.master.update()
		start9 = time.time()
		# combine SfM and MASK RCNN
		self.points3D_on_images = utils.combine(self.labeled_points, self.cameras, self.images, self.points3D)
		end9 = time.time()
		self.processing.configure(text=self.processing.cget("text")+"\nTime for combine: " + str(end9 - start9) + " seconds")
		self.master.update()
		# start5 = time.time()
		self.processed = True

		# utils.plot_points3D(self.points3D)


	def __init__(self, master):

		self.videos = ""
		self.target = ""
		self.points3D = ""
		self.processed = False
		self.project_dir = ""

		self.master = master

		self.master.title("Illegal parking report")
		self.master.geometry('800x800')

		self.frame = Frame(self.master)

		self.chosen_videos = Label(self.frame, text="Chosen videos: \n")
		def choose_clicked():
			self.videos = filedialog.askopenfilenames(parent=self.master, title='Choose videos', filetypes = (("video files","*.mp4"),("all files","*.*")))
			self.chosen_videos.configure(text="Chosen videos: \n" + ("\n").join(self.videos))
		self.chooseButton = Button(self.frame, text = 'Choose videos', width = 25, command=choose_clicked)
		self.chooseButton.pack()
		self.chosen_videos.pack()

		self.chosen_target = Label(self.frame, text="Chosen target: \n")
		def choose_clicked():
			self.target = filedialog.askopenfilename(parent=self.master, title='Choose target', filetypes = (("all files","*.*"), ("image files","*.jpg")))
			self.chosen_target.configure(text="Chosen target: \n" + str(self.target))
		self.chooseButton = Button(self.frame, text = 'Choose target', width = 25, command=choose_clicked)
		self.chooseButton.pack()
		self.chosen_target.pack()

		self.processing = Label(self.frame, text="")
		self.processButton = Button(self.frame, text = 'Process', width = 25, command=self.process_clicked)
		self.processButton.pack()
		self.processing.pack()

		self.show = Label(self.frame, text="")
		def show_3Dpoints_clicked():
			print(self.processed)
			if not self.processed:
				self.show.configure(text="ERROR!!! Please, process input")
				return
			if len(self.points3D) <= 0:
				self.cameras, self.images, self.points3D = read_model.read_model(path=self.project_dir+"sparse/0/", ext=".bin")
			self.newWindow = Toplevel(self.master)
			self.app = Show_window(self.newWindow, self.points3D)
		self.show_3DpointsButton = Button(self.frame, text = 'show 3Dpoints', width = 25, command=show_3Dpoints_clicked)
		self.show_3DpointsButton.pack()
		self.show.pack()

		self.quitButton = Button(self.frame, text = 'Quit', width = 25, command = self.close_windows)
		self.quitButton.pack()

		self.frame.pack()

class Show_window:
	def close_windows(self):
		self.master.destroy()

	def __init__(self, master, points3D):
		self.master = master
		self.frame = Frame(self.master)
		self.quitButton = Button(self.frame, text = 'Quit', width = 25, command = self.close_windows)
		self.quitButton.pack()
		self.master.wm_title("Matlabplot")

		self.fig = Figure(figsize=(5, 4), dpi=200)
		ax = self.fig.gca(projection='3d')
		ax.view_init(azim=60)

		i = 0;
		X = [];
		Y = [];
		Z = [];
		n_points = len(points3D)
		#n_points = 100
		RGB = np.zeros(shape=(n_points,3));

		for point_id in points3D:
			if i >= n_points:
				break
			point = points3D[point_id]
			xyz = point.xyz
			rgb = point.rgb  

			X.append(xyz[0])
			Y.append(xyz[1])
			Z.append(xyz[2])
			RGB[i, :] = rgb/256
			i = i+1

		Xmean = np.mean(X)
		Xstd = np.std(X)
		Ymean = np.mean(Y)
		Ystd = np.std(Y)
		Zmean = np.mean(Z)
		Zstd = np.std(Z)

		ax.scatter(X, Y, Z, facecolors = RGB, s=1)

		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')

		t_n = 2.5
		ax.set_xlim([Xmean - t_n*Xstd, Xmean + t_n*Xstd])
		ax.set_ylim([Ymean - t_n*Ystd, Ymean + t_n*Ystd])
		ax.set_zlim([Zmean - t_n*Zstd, Zmean + t_n*Zstd])

		self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)  # A tk.DrawingArea.
		self.canvas.draw()
		self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

		self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
		self.toolbar.update()
		self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
		self.frame.pack()
		
# def main():
# 	root = Tk()
# 	app = Fill_window(root)
# 	root.mainloop()

# if __name__ == "__main__":
# 	main()