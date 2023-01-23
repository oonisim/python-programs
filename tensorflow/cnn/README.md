# Cautions

image in memory passed to a most_similar() as argument MUST be BGR order as with OpenCV imread result, 
because the same transformation (resize, BGR to RGB, Keras/ResNet preprocess) is applied.
