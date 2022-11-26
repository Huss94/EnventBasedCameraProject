class Events:
    def __init__(self, x,y,t,p):

class event:
    def __init__(self, x, y, t, p):
        self.x = x
        self.y = y 
        self.t = t
        self.p = p

        self.vx = 0
        self.vy = 0
    

    def __str__(self):
        return f"(x,y,t,p) = ({self.x},{self.y},{self.t},{self.p})"