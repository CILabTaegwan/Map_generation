import os
import numpy as np
def ornot(new_list, old_list):
     for i in range(len(new_list)):
          if new_list[i] != old_list[i]:
               print(i)
     return True


import os

folder_path = 'sampling'  # sampling 폴더의 경로
file_list = os.listdir(folder_path)
file_list = [f for f in file_list if f.endswith('_seeds.npy')]  # _seeds.npy 파일만 선택

# 파일 이름에서 'th'를 제거하고 숫자만 추출하여 정렬
sorted_file_list = sorted(file_list, key=lambda x: int(x.split('_')[0].replace('th', '')))

# 정렬된 파일 리스트 출력



for file_name in sorted_file_list:
     file_path = os.path.join(folder_path, file_name)
     data = np.load(file_path)  # .npy 파일을 NumPy 배열로 읽어옴
     max_indices = np.argmax(data, axis=1)

     print(file_name,max_indices)
