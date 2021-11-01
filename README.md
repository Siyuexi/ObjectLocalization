# ObjectLocalization
All images are sampled from VID.

dataset:http://xinggangw.info/data/tiny_vid.zip
tiny_vid:
- target class: ['car', 'bird', 'turtle', 'dog', 'lizard']
- image size: 128 * 128 
- image number per class: 180
- box format in gt_XX.txt(coordinates are 0-index based): image_index, xmin, ymin, xmax, ymax

NOTE: Only one object of target classes is contained in every generated image. 