import random

def generate_data(test_points, train_points, train_cats, dimensions, testfile, trainfile):
    f = open(trainfile, "w")
    for a in range(train_cats):
        for b in range(train_points):
            line = []
            for c in range(dimensions):
                line.append("%.3f" % random.random())
            f.write(",".join(line) + "\n")
        if a != train_cats - 1: # last point in a category, leave a line
            f.write("\n")
    f.seek(f.tell()-2) # used to eliminate the last \n
    f.truncate()
    f.close()
    
    f = open(testfile, "w")
    for a in range(test_points):
        line = []
        for c in range(dimensions):
            line.append("%.3f" % random.random())
        f.write(",".join(line) + "\n")
    
    f.seek(f.tell()-2) # used to elimininate the last \n
    f.truncate()
    f.close()
    
testCases = [
    [1024, 128, 4, 64, "case1test.txt", "case1train.txt"],
    [2048, 256, 4, 64, "case2test.txt", "case2train.txt"],
    [4096, 256, 4, 64, "case3test.txt", "case3train.txt"],
    [4096, 512, 4, 64, "case4test.txt", "case4train.txt"],
    [4096, 256, 4, 128, "case5test.txt", "case5train.txt"],
    [4096, 256, 4, 256, "case6test.txt", "case6train.txt"],
    [8192, 512, 4, 256, "case7test.txt", "case7train.txt"],
    [8192, 1024, 4, 256, "case8test.txt", "case8train.txt"],
    [16384, 768, 4, 512, "case9test.txt", "case9train.txt"]
]

for case in testCases:
    generate_data(*case)
    
print("Done")