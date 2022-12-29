file_uu = open("neo0.txt", "r")
loss_li = file_uu.readline()
loss_li = [float(x) for x in loss_li.split()]

cur_times = int(file_uu.readline())

print(cur_times)