import random
from shutil import *
def get_dataset(txtpath, attribute_name):
	image_path = '/DATA3_DB7/data/yli/datasets/img_align_celeba1/'
	f = open(txtpath,'r')
	total_image_number = int(f.readline())
	attribute_name_list = f.readline().split()
	attribute_index = attribute_name_list.index(attribute_name)

	class_a = []    #none glasses; mouth close; none mustache
	class_b = []    #glasses;      mouth open;  mustache

	for i in range(total_image_number):
		image_all_attr = f.readline().split()
		image_label = [image_path+image_all_attr[0], (int(image_all_attr[attribute_index+1])+1)/2]
		if image_label[1] == 0:
			class_a.append(image_label)
		else:
			class_b.append(image_label)

	f.close()

	number_a = len(class_a)
	number_b = len(class_b)

	if attribute_name == 'Eyeglasses':
		[train_set_num, val_set_num, test_set_num] = [7900, 2600, 2693]
	elif attribute_name == 'Mouth_Slightly_Open':
		[train_set_num, val_set_num, test_set_num] = [58700, 19500, 19742]
	elif attribute_name == 'Mustache':
		[train_set_num, val_set_num, test_set_num] = [5000, 1600, 1817]

	#random.shuffle(class_a)
	train_a = class_a[0:train_set_num]
	val_a = class_a[train_set_num:train_set_num + val_set_num]
	test_a = class_a[train_set_num + val_set_num:train_set_num + val_set_num + test_set_num]

	#random.shuffle(class_b)
	train_b= class_b[0:train_set_num]
	val_b = class_b[train_set_num:train_set_num + val_set_num]
	test_b = class_b[train_set_num + val_set_num:train_set_num + val_set_num + test_set_num]
	# train = train_a + train_b
	# random.shuffle(train)
	# test = test_b + val_b
	# val = val_a + val_b
	# #random.shuffle(test)
	# random.shuffle(val)
    #
	# train_images = [x[0] for x in train]
	# train_labels = [x[1] for x in train]
    #
	# test_images = [x[0] for x in test]
	# test_labels = [x[1] for x in test]
	# val_images = [x[0] for x in val]
	# val_labels = [x[1] for x in val]
	train_A = [x[0] for x in train_a]#without attribute
	train_B = [x[0] for x in train_b]#with attribute
	val_A = [x[0] for x in val_a]
	val_B = [x[0] for x in val_b]
	test_A = [x[0] for x in test_a] + val_A
	test_B = [x[0] for x in test_b] + val_B
	test_B_label = [x[1] for x in test_b]+[x[1] for x in val_b]
	#print(test_B_label)

	return train_A, train_B, test_A, test_B
if __name__ == '__main__':
	save_path = '/DATA3_DB7/data/yli/datasets/'
	_,_, test_path,new = get_dataset(save_path+'list_attr_celeba.txt','Eyeglasses')
	print(len(new))
	for path in new:
		copyfile(path,'./test/'+path[-10:])