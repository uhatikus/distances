import os
import cv2
import ntpath
import sys
import read_model


def video_to_frames(video_file):
	vidcap = cv2.VideoCapture(video_file)
	video_name = ((ntpath.basename(video_file)).split("."))[0]
	os.system("mkdir data/projects/" + video_name)
	os.system("mkdir data/projects/" + video_name + "/images")
	success,image = vidcap.read()
	count = 0
	project_dir = os.getcwd() + "/data/projects/" + video_name + "/"
	while success:
		cv2.imwrite(project_dir + "images/" + video_name + "_frame_%d.jpg" % count, image)     # save frame as JPEG file      
		vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100))
		success,image = vidcap.read()
		print('Read a new frame: ', success)
		count += 1	
		if count >= 0:
			break
	return project_dir


def main():
	if len(sys.argv) != 2:
		print("Usage: python3 main.py path/to/video_file")
		return

	video_file = sys.argv[1]
	project_dir = video_to_frames(video_file)
	#os.system("colmap automatic_reconstructor --workspace_path " + project_dir + " --image_path " + project_dir + "images/")
	#os.system("colmap feature_extractor  --database_path " + project_dir + "/database.db --image_path " + project_dir + "/images")
	
	cameras, images, points3D = read_model.read_model(path=project_dir+"sparse/0/", ext=".bin")
	
	#read_model.plot_points3D(points3D)

if __name__ == "__main__":
    main()
