from PIL import Image
im = Image.open("picture/obm2.jpg")
bg = Image.new("RGB", im.size, (255,255,255))
bg.paste(im,im)
bg.save(r"obm2.jpg")