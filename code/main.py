import sys
import read_model
import utils


def main():
	if len(sys.argv) != 2:
	 	print("Usage: python3 main.py path/to/video_file")
		# return

	project_dir = utils.video_to_frames(sys.argv[1])
	
	# os.system("colmap automatic_reconstructor --workspace_path " + project_dir + " --image_path " + project_dir + "images/")
	# os.system("colmap feature_extractor  --database_path " + project_dir + "/database.db --image_path " + project_dir + "/images")
	
	cameras, images, points3D = read_model.read_model(path=project_dir+"sparse/0/", ext=".bin")


	# utils.plot_points3D(points3D)


	f = open("/Users/ukhatov/Documents/Projects/distances/code/points.txt", 'r')
	points3D_on_images = []
	for line in f.readlines():
		if ']' in line:
			ent = line.split()
			for i in ent:
				if i[-1] == ']':
					points3D_on_images.append(int(i[0:-1]))
	# print(points3D_on_images)
	s = "0 1 1 1 1 0 1"
	labels_ = s.split()
	print(len(labels_))
	id_num = 0

	mark_list = []
	count__ = 0
	for i, label in enumerate(labels_):
	    if int(label) == id_num:
	    # if True:
	    	count__ += 1
	    	mark_list.append(points3D_on_images[i])
	print(count__)
	utils.plot_points3D_after(points3D, mark_list)
	#labeled_points = utils.label_points(project_dir + "images/")
	
	#points3D_on_images = utils.combine(labeled_points, cameras, images, points3D)


if __name__ == "__main__":
    main()
