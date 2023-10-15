#pip install pillow-heif
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()
start_idx = 1328
end_idx = 1338
max_decimal_order = 3
n = 0

for i in range(start_idx, end_idx):
    image = Image.open("iphone_sample_images/IMG_"+str(i)+".HEIC")
    #newsize = (4032//8,3024//8)
    #image = image.resize(newsize)
    print("extracted ",i)

    str_code = str(n)
    strc = ""
    for j in range( max_decimal_order-len(str_code) ) :
        strc = strc+"0"
    strc = strc+str_code
    print("got strc ",strc)

    #image.save("extracted/"+strc+".png",format="png")
    image.save("extracted/"+strc+".jpg",format="jpeg")

    n+=1