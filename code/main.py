import sys
import read_model
import utils


def main():
	if len(sys.argv) != 2:
		print("Usage: python3 main.py path/to/video_file")
		return

	project_dir = utils.video_to_frames(sys.argv[1])
	
	os.system("colmap automatic_reconstructor --workspace_path " + project_dir + " --image_path " + project_dir + "images/")
	os.system("colmap feature_extractor  --database_path " + project_dir + "/database.db --image_path " + project_dir + "/images")
	
	cameras, images, points3D = read_model.read_model(path=project_dir+"sparse/0/", ext=".bin")
	
	# utils.plot_points3D(points3D)
	labeled_points = utils.label_points(project_dir + "images/")
	# print(labeled_points)


if __name__ == "__main__":
    main()
