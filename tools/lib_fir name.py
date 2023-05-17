import os
import string


def reName(dirname):
   count = 0
   for cur_file in os.listdir(dirname):
     count = count+1
     oldDir = os.path.join(dirname, cur_file)
     filetype = os.path.splitext(cur_file)[1] # 文件类型
     if(count<10):
         newDir = os.path.join(dirname,'0000' + str(count) + filetype)
         os.rename(oldDir, newDir)
     if(count<100):
         newDir = os.path.join(dirname, '000' + str(count) + filetype)
         os.rename(oldDir, newDir)
     if (count < 1000):
         newDir = os.path.join(dirname, '00' + str(count) + filetype)
         os.rename(oldDir, newDir)
     if (count < 10000):
         newDir = os.path.join(dirname, '0' + str(count) + filetype)
         os.rename(oldDir, newDir)
     print(oldDir, newDir)


if __name__ == "__main__":
 dirname = r"../data/class/risehand_"#文件夹路径
 reName(dirname)