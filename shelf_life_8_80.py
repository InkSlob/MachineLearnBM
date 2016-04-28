import pickle

# OPEN ShelfLife Pickle File
ShelfLife=[]
#fShelfLife = open('/home/master/Documents/Blmbrg/Ln_Rg_Model/ShelfLife_15k_Stemmed.txt', 'rb') 
fShelfLife = open('ShelfLife_15k_Stemmed.txt', 'rb') 
ShelfLife = pickle.load(fShelfLife)
#check if pickle loaded properly
if(len(ShelfLife) == 15000):
    print "ShelfLife pickle file has loaded with proper size of 15,000.  Size = ", len(ShelfLife), "\n"

cnt = 0
Shelf_Life_8_80=[]
for i in range(0, len(ShelfLife)-1):
    tmp = ShelfLife[i]
    print "cnt: ", i
    if (tmp <= 8):
        Shelf_Life_8_80.append(1)
    elif (tmp >= 80):
        Shelf_Life_8_80.append(2)
    else:
        Shelf_Life_8_80.append(0)

        
        
for x in range(1, 25):
    print Shelf_Life_8_80[x]
