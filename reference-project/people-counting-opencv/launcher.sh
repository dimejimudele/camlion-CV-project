python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
	--input videos/example_01.mp4 --output output/output_02.avi 
	python people_counter.py --prototxt yolov3-tiny.cfg -- model yolov3-tiny.weights --input videos/e1.mp4 
