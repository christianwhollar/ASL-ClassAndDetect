import json
with open('classes.json', 'r') as f:
        classes = json.load(f)
        
Categories=classes.keys()

flat_data_arr=[] #input array
target_arr=[] #output array

datadir='IMAGES/' 

#path which contains all the categories of images
for i in Categories:
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
    
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)