from pyspark import SparkContext

logFile = r'C:\spark\spark-2.2.0-bin-hadoop2.7\README.md'
logdir = r'C:\spark\spark-2.2.0-bin-hadoop2.7\*.txt'
logdir1 = r'C:\spark\spark-2.2.0-bin-hadoop2.7'
sc = SparkContext(master="local[2]",appName="Simple App")


rdd = sc.parallelize([('a',7),('a',2),('b',2)])
rdd3 = sc.parallelize(range(100))
data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data,)
a = distData.reduce(lambda a, b: a + b)

print(a)

# print(a)
#

def map_function(line):
    return len(line.split(' '))
logData1 = sc.textFile(logFile).cache()
logData = sc.textFile(logFile)
logData2 = sc.textFile(logdir) # đọc nhiều file thì sẽ map các file lại thành 1 ( map các line lại với nhau)
logData3 = sc.wholeTextFiles(logdir1) # đọc nhiều file thì sẽ map các file lại thành 1 ( map các line lại với nhau)


numAs = logData.filter(lambda line: 'a' in line).count()
numbBs = logData.filter(lambda line: 'b' in line).count()
a = logData.map(lambda x: len(x.split())).reduce(lambda a,b: a+b)
word_list = logData.flatMap(lambda x: x.split()) # tạo ra 1 RDD chứa data là các word, trong file
word_list1 = logData.map(lambda x: x.split()) # tạo ra 1 RDD chứa data là các word, trong file


map_for_reduce = word_list.map(lambda x: (x, 1))
word_count = map_for_reduce.reduceByKey(lambda a, b: a + b)
word_count = word_list.map(lambda x: (x,1)).reduceByKey(lambda a,b: a+b)
word_count_tuple = map_for_reduce.reduce(lambda a, b: a + b)
map_for_reduce.persist() # save map_for_reduce after first time computed, later use map_for_reduce don't need compute again.

map_1 = logData.map(map_function).reduce(lambda a,b: a if a >b else b)

counter = 0
rdd = sc.parallelize(data)

# Wrong: Don't do this!!
def increment_counter(x):
    global counter
    counter += x
rdd.foreach(increment_counter)

print("Counter value: ", counter)

x = logData.filter(lambda x: 'a' in x)




sc.stop()

rdd3.groupBy(lambda x: x % 2).mapValues(list).collect()