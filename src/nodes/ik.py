import math

def ik(x, y):
    if x == 0 and y == 0:
        x = 90 
        y = 40

    if x < 90:
        return 0, 0, False

    if x < 180 and y < 80:
        return 0, 0, False

    if y < -20:
        return 0, 0, False

    h1=120
    h2=122.5
    l3=math.sqrt(x*x+y*y)
    num=h1*h1-h2*h2+l3*l3
    den=2*h1*l3

    if num/den>1:
        theta1=0
        ja=200
        jb=200
    elif num/den<-1:
        theta1=0
        ja=0
        jb=0
    else:
        cosb=(h1*h1-h2*h2+l3*l3)/(2*h1*l3);	
        jb=math.acos(cosb)
        ja=math.atan2(y,x)
        theta1=ja+jb

    has_solution = True
    if ((x - h1*math.cos(theta1))/h2)*((x - h1*math.cos(theta1))/h2)<1: 
        if h1*math.sin(theta1)>y:
            theta2 = -math.acos((x - h1*math.cos(theta1))/h2) - theta1
        else:
            theta2 = math.acos((x - h1*math.cos(theta1))/h2) - theta1
    else:
        theta2=0
        has_solution=False

    return theta1,theta2, has_solution
