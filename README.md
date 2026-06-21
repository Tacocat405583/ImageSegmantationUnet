THIS IS A BASIC UNET SEGMANTATION MODEL

I would recommend using uv library to run this, then running uv sync to sintall all dependencies

test.py and train.py include commented out global variables, MAKE SURE TO READ OVER ONCE YOU KNOW WHICH GPU YOU HAVE 

With current parameters, cpu training will take almost 40 minutes with a 9800X3D as a benchmark.
I would recommend lowering parameters without a GPU to train on or use GoogleColab

When running test.py after training your model, predictions will be replaced with your new segmantation predictions, so save
your data incase you want to compare.


UNET CODE credit given to @aladdinpersson from his video https://www.youtube.com/watch?v=IHq1t7NxS8k&list=LL&index=23&t=1811s
