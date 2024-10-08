from PIL import Image, ImageOps
import re

def tryfloat(s):
    try:
        return float(s)
    except:
        return s

def alphanum_key(s):
    return [ tryfloat(c) for c in re.split('(-*\d+\.\d*)' , s) ]

def image_grid(imgs, rows, cols, side=256, sign_sort=True):
    assert len(imgs) == rows*cols
    
    if sign_sort:
        imgs.sort(key=alphanum_key)

    grid = Image.new('RGB', size=(cols*side, rows*side))
    
    for i, img in enumerate(imgs):
        add_frame = (True if "start1.0_incre0.00" in img else False)
        img = Image.open(img).resize((side,side))
        if add_frame:
            img = ImageOps.expand(img, border=20, fill='white').resize((side,side))
        grid.paste(img, box=(i//rows*side, i%rows*side))
    return grid
