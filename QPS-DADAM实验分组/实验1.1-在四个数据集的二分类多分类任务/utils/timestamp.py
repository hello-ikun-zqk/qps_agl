import time

def get_strftime():
    now = int(round(time.time()*1000))
    now = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(now/1000))
    return now


if __name__=="__main__":
    print(get_strftime())