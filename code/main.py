from tkinter import *
import interface

def main(): 
	root = Tk()
	app = interface.Fill_window(root)
	root.mainloop()

if __name__ == "__main__":
	main()

# def main():
# 	len_args = len(sys.argv)
# 	video_files = []
# 	if len(sys.argv) < 2:
# 		print("Usage: python3 code/main.py path/to/video_file_1 path/to/video_file_2 ... Default videos will be used")
# 		video_files.append("/Users/ukhatov/Documents/Projects/distances/data/videos/20190808_151553.mp4")
# 		video_files.append("/Users/ukhatov/Documents/Projects/distances/data/videos/20190808_151758.mp4")
# 	else:
# 		for i in range(len_args-1):
# 			video_files.append(sys.argv[i+1])

# 	start1 = time.time()
# 	project_dir, n_imgs = utils.video_to_frames(video_files)
# 	end1 = time.time()
# 	print("time for video_to_frames: " + str(end1 - start1) + " seconds")
# 	start2 = time.time()
# 	# get model for labeling 
# 	model = utils.get_model(project_dir + "images/", n_imgs)
# 	end2 = time.time()
# 	print("time for get_model: " + str(end2 - start2) + " seconds")
# 	start3 = time.time()
# 	# labeling for cropping
# 	labeled_points_for_cropping = utils.label_points(project_dir + "images/", model)
# 	end3 = time.time()
# 	print("time for label_points: " + str(end3 - start3) + " seconds")
# 	start4 = time.time()
# 	# saved in directory "cropped"
# 	utils.crop_frames(labeled_points_for_cropping, project_dir, "cropped", "images")
# 	end4 = time.time()
# 	print("time for crop_frames: " + str(end4 - start4) + " seconds")
# 	start5 = time.time()
# 	# get target frames from the cropped using feature extraction 
# 	utils.get_target_frames(project_dir, "target.jpg", "target")
# 	end5 = time.time()
# 	print("time for get_target_frames: " + str(end5 - start5) + " seconds")
	
# 	# os.system("colmap automatic_reconstructor --workspace_path " + project_dir + " --image_path " + project_dir + "target/")
	
# 	# cameras, images, points3D = read_model.read_model(path=project_dir+"sparse/0/", ext=".bin")

# 	# utils.plot_points3D(points3D)

# 	# labeled_points = utils.label_points(project_dir + "target/")
	
# 	# points3D_on_images = utils.combine(labeled_points, cameras, images, points3D)


# if __name__ == "__main__":
#     main()
