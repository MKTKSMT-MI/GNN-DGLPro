import csv
import os


a = [r for r in range((9 + 39))]
b = [str(r) for r in a]

with open('test.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(a)


print(os.path.getsize('test.txt'))
print(os.path.getsize('test.csv'))
