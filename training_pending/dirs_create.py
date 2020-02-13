# Used to find difference between two frames

# This is to create the folders in mapping
# after executing this, run diff_cms in home by following instruction in notes
# Timing - around 30 seconds (2 speakers)
import os

mapping_path = "/home/arasu/FYP/code/sample/mapping"
diff_path = "/home/arasu/FYP/code/sample/keyframes"

obj1 = os.scandir(mapping_path)

l = ""

for entry1 in obj1 :
    if entry1.is_dir():
        obj2 = os.scandir(os.path.join(mapping_path, entry1.name))
        for entry2 in obj2 :
            if entry2.is_dir():
                l += "mkdir -p '{}'".format(os.path.join(diff_path, entry1.name, entry2.name)) + '\n'

with open("dirs_create.sh",'w') as f:
    f.write(l)
    print("Done writing cmds to file")
    
print("Please run dirs_create.sh now")
