import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
accuracy =[]
accuracy[1]=70.488
accuracy[3]=81.975
accuracy[5]=80.226
accuracy[7]=80.661
plt.ioff()
plt.gcf()
plt.plot(accuracy, label="test accuracy")
plt.legend(loc='upper right')
plt.xlabel('Number of layers')
plt.ylabel('Test accuracy[%]')
