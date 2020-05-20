import cv2
import numpy as np
import sys
import math

class Delaunay2D:
    def find_circumcenter(self, tri):
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                      [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        radius = np.sum(np.square(pts[0] - center))
        return (center, radius)

    def inCircleFast(self, tri, p):
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def addCoordinate(self, p):
        p = np.asarray(p)
        idx = len(self.coords)
        self.coords.append(p)
        bad_triangles = []
        for T in self.triangles:
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        boundary = []
        arr = []
        T = bad_triangles[0]
        edge = 0
        while True:
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))
                edge = (edge + 1) % 3
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            T = (idx, e0, e1)
            self.circles[T] = self.find_circumcenter(T)
            self.triangles[T] = [tri_op, None, None]

            if tri_op:
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            self.triangles[tri_op][i] = T

            new_triangles.append(T)
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]
            self.triangles[T][2] = new_triangles[(i-1) % N]

    def findTriangles(self):
        return [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]
    def __init__(self, center=(0, 0), radius=9999):
        center = np.asarray(center)
        self.coords = [center+radius*np.array((-1, -1)),
                       center+radius*np.array((+1, -1)),
                       center+radius*np.array((+1, +1)),
                       center+radius*np.array((-1, +1))]

        self.triangles = {}
        self.circles = {}

        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        for t in self.triangles:
            self.circles[t] = self.find_circumcenter(t)



#area of Triangle
def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


#checks if point (x, y) lies inside Triangle formed by (x1, y1), (x2, y2), (x3, y3)
def isInside(x1, y1, x2, y2, x3, y3, x, y):
    A  = area (x1, y1, x2, y2, x3, y3)
    A1 = area (x, y, x2, y2, x3, y3)
    A2 = area (x1, y1, x, y, x3, y3)
    A3 = area (x1, y1, x2, y2, x, y)
    if(A == A1 + A2 + A3):
        return 1
    else:
        return 0
#maps triangles on all k frames
def form_destination_triangle(img,color,source,triplets) :
    for i in range(len(target)):
        pt1 = source[target[i][0]]
        pt2 = source[target[i][1]]
        pt3 = source[target[i][2]]
        cv2.line(img, pt1, pt2, color,  2, 2)
        cv2.line(img, pt2, pt3, color,  2, 2)
        cv2.line(img, pt3, pt1, color,  2, 2)
        triplets.append((pt1,pt2,pt3))
#does triangulation on first frame which is then mapped to all k frames
def draw_delaunay(img, delaunay_color, src_coordinates) :

    size = img.shape
    r = (0, 0, size[1], size[0])
    delaunay = Delaunay2D()
    for i in src_coordinates:
        delaunay.addCoordinate(i)
    triangleList = delaunay.findTriangles()
    for t in triangleList :
        # pt1 = (t[0], t[1])
        # pt2 = (t[2], t[3])
        # pt3 = (t[4], t[5])
        pt1 = (src_coordinates[t[0]])
        pt2 = (src_coordinates[t[1]])
        pt3 = (src_coordinates[t[2]])

        cv2.line(img, pt1, pt2, delaunay_color,  2, 2)
        cv2.line(img, pt2, pt3, delaunay_color,  2, 2)
        cv2.line(img, pt3, pt1, delaunay_color,  2, 2)
        src_triplets.append((pt1,pt2,pt3))
        for i in range(len(src_coordinates)):
            if(src_coordinates[i] == pt1):
                x = i
            if(src_coordinates[i] == pt2):
                y = i
            if(src_coordinates[i] == pt3):
                z = i
        target.append((x,y,z))

def draw_delaunay_hardcode():
    triangleList = [(0, 1, 4), (0, 4, 2), (4, 5, 2), (5, 2, 3), (5, 6, 3), (6, 1, 3), (1, 4, 6), (4, 5, 6)]
    for t in triangleList :
        # pt1 = (t[0], t[1])
        # pt2 = (t[2], t[3])
        # pt3 = (t[4], t[5])
        pt1 = (src_coordinates[t[0]])
        pt2 = (src_coordinates[t[1]])
        pt3 = (src_coordinates[t[2]])

        cv2.line(img, pt1, pt2, delaunay_color,  2, 2)
        cv2.line(img, pt2, pt3, delaunay_color,  2, 2)
        cv2.line(img, pt3, pt1, delaunay_color,  2, 2)
        src_triplets.append((pt1,pt2,pt3))
        for i in range(len(src_coordinates)):
            if(src_coordinates[i] == pt1):
                x = i
            if(src_coordinates[i] == pt2):
                y = i
            if(src_coordinates[i] == pt3):
                z = i
        target.append((x,y,z))


#select points from source image
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,', ',y)
        src_coordinates.append((x,y))

#select points from destination image
def click_event1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,', ',y)
        dest_coordinates.append((x,y))


#read images
src_img = cv2.imread("Bush.jpg");
dest_img = cv2.imread("Clinton.jpg");
#stores control points
src_coordinates = []
dest_coordinates = []
#stores triangle coordinates
src_triplets = []
dest_triplets = []
#store triangles indices used in mapping
target = []
#corner points

src_img = cv2.resize(src_img, (500, 500))
dest_img = cv2.resize(dest_img, (500, 500))
src_coordinates.append((0,0))
src_coordinates.append((0,499))
src_coordinates.append((499,0))
src_coordinates.append((499,499))

dest_coordinates.append((0,0))
dest_coordinates.append((0,499))
dest_coordinates.append((499,0))
dest_coordinates.append((499,499))

#use mouse click to get points
cv2.imshow('image',src_img)

cv2.setMouseCallback('image',click_event)

cv2.waitKey(0)

cv2.imshow('image',dest_img)

cv2.setMouseCallback('image',click_event1)

cv2.waitKey(0)

img_cpy = src_img.copy();

#draw delaunay Trianglulation on src image
draw_delaunay(img_cpy, (0,0,255), src_coordinates);

cv2.imshow('image',img_cpy)
cv2.waitKey(0)

img_cpy_dest = dest_img.copy();
img_cpy_src_2 = src_img.copy();

src_triplets_cpy = []

#draw delaunay Trianglulation on src and destination image
form_destination_triangle(img_cpy_dest, (0,0,255), dest_coordinates, dest_triplets);
form_destination_triangle(img_cpy_src_2, (0,0,255), src_coordinates, src_triplets_cpy);

cv2.imshow('image',img_cpy_dest)
cv2.waitKey(0)

cv2.imshow('image',img_cpy_src_2)
cv2.waitKey(0)

# set N and i  = number of frames
N = 20
frames = []
i = 20

while i>0:
    img_1 = np.zeros([500,500,3],dtype=np.uint8)
    frames.append(img_1)
    i = i-1


intermediate_coordinates = []

#linear interpolation of control points
for i in range(len(frames)):
    coordi = []
    for j in range(len(dest_coordinates)):
        y = src_coordinates[j]
        z = dest_coordinates[j]

        x1 = ((N - i - 1) / N) * y[0]
        y1 = ((N - i - 1) / N) * y[1]
        x2 = ((i + 1) / N) * z[0]
        y2 = ((i + 1) / N) * z[1]
        x1 = round(x1+x2)
        y1 = round(y1+y2)
        x = (x1, y1)
        coordi.append(x)

    intermediate_coordinates.append(coordi)

frame_triplets = []

#triangulate k frames
for i in range(len(frames)):
    trrip = []
    form_destination_triangle(frames[i],(255,255,255),intermediate_coordinates[i],trrip)
    frame_triplets.append(trrip)

#assigning color to remaining points using affine basis
for i in range(len(frames)):
    for j in range(0,500):
        for k in range(0,500):
            for l in range(len(frame_triplets[i])):
                if isInside(frame_triplets[i][l][0][0],frame_triplets[i][l][0][1],frame_triplets[i][l][1][0],frame_triplets[i][l][1][1],frame_triplets[i][l][2][0],frame_triplets[i][l][2][1],j,k) == 1:
                    index = l
                    #calculate alpha and beta from given Points
                    alpha = ((frame_triplets[i][l][2][0]-frame_triplets[i][l][0][0])*(k-frame_triplets[i][l][0][1]) - (j-frame_triplets[i][l][0][0])*(frame_triplets[i][l][2][1]-frame_triplets[i][l][0][1]))
                    alpha = alpha/((frame_triplets[i][l][2][0]-frame_triplets[i][l][0][0])*(frame_triplets[i][l][1][1]-frame_triplets[i][l][0][1]) - (frame_triplets[i][l][1][0]-frame_triplets[i][l][0][0])*(frame_triplets[i][l][2][1]-frame_triplets[i][l][0][1]))

                    beta = ((frame_triplets[i][l][1][0]-frame_triplets[i][l][0][0])*(k-frame_triplets[i][l][0][1]) - (j-frame_triplets[i][l][0][0])*(frame_triplets[i][l][1][1]-frame_triplets[i][l][0][1]))
                    beta = beta/((frame_triplets[i][l][1][0]-frame_triplets[i][l][0][0])*(frame_triplets[i][l][2][1]-frame_triplets[i][l][0][1]) - (frame_triplets[i][l][2][0]-frame_triplets[i][l][0][0])*(frame_triplets[i][l][1][1]-frame_triplets[i][l][0][1]))
                    #use alpha beta to find point in kth frame and then assign a color
                    x_src = alpha*(src_triplets_cpy[l][1][0]-src_triplets_cpy[l][0][0]) + beta*(src_triplets_cpy[l][2][0]-src_triplets_cpy[l][0][0]) + src_triplets_cpy[l][0][0]
                    y_src = alpha*(src_triplets_cpy[l][1][1]-src_triplets_cpy[l][0][1]) + beta*(src_triplets_cpy[l][2][1]-src_triplets_cpy[l][0][1]) + src_triplets_cpy[l][0][1]


                    x_dest = alpha*(dest_triplets[l][1][0]-dest_triplets[l][0][0]) + beta*(dest_triplets[l][2][0]-dest_triplets[l][0][0]) + dest_triplets[l][0][0]
                    y_dest = alpha*(dest_triplets[l][1][1]-dest_triplets[l][0][1]) + beta*(dest_triplets[l][2][1]-dest_triplets[l][0][1]) + dest_triplets[l][0][1]
                    x_dest = round(x_dest)
                    y_dest = round(y_dest)
                    x_src = round(x_src)
                    y_src = round(y_src)
                    #assign colors using the linear interpolation of src and destination image
                    for z in range(0,3):
                        x1 = round(((N - i - 1) / N) * src_img[x_src][y_src][z])
                        x2 = round(((i + 1) / N) * dest_img[x_dest][y_dest][z])
                        frames[i][j][k][z] = (x1 + x2)

                    break
    #saving all frames

    names = "frame" + str(i) + ".jpg"
    cv2.imwrite(names,frames[i])
