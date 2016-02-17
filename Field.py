import numpy as np
import sys, time
import curses


class Field(object):
    """
    Field object
    """
    size = np.array([10,20])

    def __init__(self, destination, current=None):
        if current:
            self.crt = np.array(current)
        else:
            self.crt = self.setinitialposition
        self.dst = np.array(destination)
        self.arraynum =np.zeros(self.size)
        self.setarraystr()

    def setinitialposition(self):
        x = np.random.randn(2)
        return np.floor(x * self.size).astype(np.int16)

    def update(self, nextpos):
        self.crt = nextpos

    def setarraystr(self):
        L = []
        L.append("\r|")
        for i in range(0, self.size[0] + 2):
            if i == 0 or i == self.size[0] + 1:
                for j in range(0, self.size[1]):
                    L.append("-")
                L.append("\x1b[0m|\n\r")
                L.append("\x1b[0m|")
            else:
                for j in range(0, self.size[1]):
                    L.append(" ")
                L.append("\x1b[0m|\n\r")
                L.append("\x1b[0m|")

        crtindex = self.crt[0] * (self.size[1] + 2) + self.crt[1]
        dstindex = self.dst[0] * (self.size[1] + 2) + self.dst[1]

        L[crtindex] = "\x1b[0m*"
        L[dstindex] = "\x1b[1;31m*"
        self.arraystr = ''.join(L)

    def eachline(self):
        L = []
        L.append("|")
        for i in range(0, self.size[0] + 2):
            if i == 0:
                for j in range(0, self.size[1]):
                    L.append("-")
                L.append("|")
            elif i == self.size[0] + 1:
                L.append("|")
                for j in range(0, self.size[1]):
                    L.append("-")
                L.append("|")
            else:
                L.append("|")
                for j in range(0, self.size[1]):
                    L.append(" ")
                L.append("|")

        crtindex = self.crt[0] * (self.size[1] + 2) + self.crt[1]
        dstindex = self.dst[0] * (self.size[1] + 2) + self.dst[1]

        L[crtindex] = "*"
        L[dstindex] = "*"

        M = []
        for i in range(0, self.size[0] + 2):
            M.append(''.join(L[ (self.size[1] + 2) * i:(self.size[1] + 2) * (i + 1)]))
        return M

    def visualize(self):
        self.setarraystr()
        sys.stdout.write(self.arraystr)
        sys.stdout.flush()
        time.sleep(0.0001)


# print("*\x1b[1;31m*\x1b[0m*", "\r", , end="")


class Q(Field):
    pass


if __name__ == "__main__":
    c = Field([2,2], [8,16])
#    c.visualize()

    print(c.eachline())
"""
    for i in range(0, 10):
        x = np.floor(np.random.randn(1) * c.size).astype(np.int16)
        c.update(x)
        sys.stdout.write(c.arraystr)
#        c.visualize()
"""
"""
import sys, time

c = Field([2,2], [8,16])
c.visualize()
for num, i in enumerate(range(10)):
    sys.stdout.write(c.arraystr)
    sys.stdout.flush()
    time.sleep(0.01)


"""
"""
import numpy as np
import sys, time
import curses
from Field import Field


c = Field([2,2], [8,16])

stdscr = curses.initscr()
curses.noecho()
curses.cbreak()

for i in range(0, 10):
    x = np.floor(np.random.randn(1) * c.size).astype(np.int16)
    c.update(x)
    m = c.eachline()
    for j in range(len(m)):
        stdscr.addstr(j, 0, m[j])
    stdscr.refresh()
    time.sleep(0.5)

curses.echo()
curses.nocbreak()
curses.endwin()


"""

