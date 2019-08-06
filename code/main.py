import sys
import read_model
import utils




def main():
	len_args = len(sys.argv)
	video_files = []
	if len(sys.argv) < 2:
		print("Usage: python3 code/main.py path/to/video_file_1 path/to/video_file_2 ... Default videos will be used")
		video_files.append("/Users/ukhatov/Documents/Projects/distances/data/videos/20190805_155841.mp4")
		video_files.append("/Users/ukhatov/Documents/Projects/distances/data/videos/20190805_155723.mp4")
	else:
		for i in range(len_args-1):
			video_files.append(sys.argv[i+1])

	project_dir = utils.video_to_frames(video_files)
	
	# get model for labeling 
	model = utils.get_model(project_dir + "images/")
	# labeling for cropping
	labeled_points_for_cropping = utils.label_points(project_dir + "images/", model)

	# saved in directory "cropped"
	utils.crop_frames(labeled_points_for_cropping, project_dir, "cropped", "images")

	# get target frames from the cropped using feature extraction 
	utils.get_target_frames(project_dir, "target.jpg", "target")

	os.system("colmap automatic_reconstructor --workspace_path " + project_dir + " --image_path " + project_dir + "target/")
	
	cameras, images, points3D = read_model.read_model(path=project_dir+"sparse/0/", ext=".bin")

	utils.plot_points3D(points3D)

	labeled_points = utils.label_points(project_dir + "target/")
	
	points3D_on_images = utils.combine(labeled_points, cameras, images, points3D)


if __name__ == "__main__":
    main()
