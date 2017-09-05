import xlrd
import numpy as np

# data = xlrd.open_workbook('/home/wangfeng/PysicalProperty_human.xlsx')
# table = data.sheets()[0]
# nrows = table.nrows
# ncols = table.ncols
# class_attribute = []
# for i in xrange(1,nrows):
#     rowValues= table.row_values(i)
#     class_attribute.append(rowValues[1:])
# np.save('class_attribute_matrix', class_attribute)
a = np.load('data/class_attribute_matrix.npy')
print a