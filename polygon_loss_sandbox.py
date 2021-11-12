import torch
import numpy as np
import cv2
from scipy.spatial import ConvexHull

def raster_poly_IOU(poly1,poly2,scale = 1000):
    poly1 = (poly1 * scale/2.0 + scale/4.0).detach().numpy().astype(np.int32)
    poly2 = (poly2 * scale/2.0 + scale/4.0).detach().numpy().astype(np.int32)
    
    im1 = np.zeros([scale,scale])
    im2 = np.zeros([scale,scale])
    imi = np.zeros([scale,scale])
    
    im1 = cv2.fillPoly(im1,[poly1],color = 1)
    im2 = cv2.fillPoly(im2,[poly2],color = 1)
    imi = (im1 + im2)/2.0

    cv2.imshow("imi",imi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    imi = np.floor(imi)
    
    ai = np.sum(imi)
    a1 = np.sum(im1)
    a2 = np.sum(im2)
    
    iou = (ai)/(a1 + a2 - ai)
    
    return iou

def get_poly(starting_points = 10):
    test = torch.rand([starting_points,2],requires_grad = False)
    test = get_hull(test)
    return test

def get_hull(points,indices = False):
    hull = ConvexHull(points.clone().detach()).vertices.astype(int)
    
    if indices:
        return hull
    
    points = points[hull,:]
    return points

def poly_area(polygon):
    """
    Returns the area of the polygon
    polygon - [n_vertices,2] tensor of clockwise points
    """
    x1 = polygon[:,0]
    y1 = polygon[:,1]
    
    x2 = x1.roll(1)
    y2 = y1.roll(1)
    
    # per this formula: http://www.mathwords.com/a/area_convex_polygon.htm
    area = -1/2.0 * (torch.sum(x1*y2) - torch.sum(x2*y1))
    
    return area

def plot_poly(poly,color = (0,0,255),im = None,lines = True,show = True,text = None):
    
    if im is None:
        s = 1000
        im = np.zeros([s,s,3]) + 255
    else:
        s = im.shape[0]
    
    if len(poly) > 0:
        poly = poly * s/2.0 + s / 4.0
    
    for p_idx,point in enumerate(poly):
        point = point.int()
        im = cv2.circle(im,(point[0],point[1]),3,color,-1)
        
    if lines:
        for i in range(-1,len(poly)-1):
            p1 = poly[i].int()
            p2 = poly[i+1].int()
            im = cv2.line(im,(p1[0],p1[1]),(p2[0],p2[1]),color,1)
    
    if text is not None:    
        im = cv2.putText(im,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1)

    if show:
        cv2.imshow("im",im)
        cv2.waitKey(1000)
        #cv2.destroyAllWindows()
    return im

def plot_adjacency(points,adjacency,color = (0,0,255),im = None):
    if im is None:
        s = 1000
        im = np.zeros([s,s,3]) + 255
    else:
        s = im.shape[0]
    
    if len(points) > 0:
        points = points * s/2.0 + s / 4.0
    
    for i in range(adjacency.shape[0]):
        p1 = points[i,:].int()
        
        for j in range(adjacency.shape[1]):
            if adjacency[i,j] == 1:
                p2 = points[j,:].int()
                im = cv2.line(im,(p1[0],p1[1]),(p2[0],p2[1]),color,2)  
        im = cv2.putText(im,str(i),(p1[0],p1[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1)

    
    cv2.imshow("im",im)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return im

def do_something(poly1,poly2):
    """
    Calculate the intersection over union between two convex polygons
    poly1,poly2 - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    """
    
    # blank image
    im = np.zeros([1000,1000,3]) + 255
    
    # for each polygon, sort vertices in a clockwise ordering
    poly1 = clockify(poly1)
    poly2 = clockify(poly2)
    
    # plot the polygons
    im = plot_poly(poly1,color = (0,0,255),im = im,show = False)
    im = plot_poly(poly2,color = (255,0,0),im = im,show = False)

    # find all intersection points between the two polygons - needs to be differentiable
    # we follow this formulation: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    
    
    # tensors such that elementwise each index corresponds to the interstection of a poly1 line and poly2 line
    xy1 = poly1.unsqueeze(1).expand([-1,poly2.shape[0],-1])
    xy3 = poly2.unsqueeze(0).expand([poly1.shape[0],-1,-1])
    
    # same data, but rolled to next element
    xy2 = poly1.roll(1,0).unsqueeze(1).expand([-1,poly2.shape[0],-1])
    xy4 = poly2.roll(1,0).unsqueeze(0).expand([poly1.shape[0],-1,-1])
    
    
    x1 = xy1[:,:,0]
    y1 = xy1[:,:,1]
    x2 = xy2[:,:,0]
    y2 = xy2[:,:,1]
    x3 = xy3[:,:,0]
    y3 = xy3[:,:,1]
    x4 = xy4[:,:,0]
    y4 = xy4[:,:,1]
    
    # Nx and Ny contain x and y intersection coordinates for each pair of line segments
    D = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    Nx = ((x1*y2 - x2*y1)*(x3-x4) - (x3*y4 - x4*y3)*(x1-x2)) / D
    Ny = ((x1*y2 - x2*y1)*(y3-y4) - (x3*y4 - x4*y3)*(y1-y2)) / D
    
    # get points that intersect in valid range (Nx should be greater than exactly one of x1,x2 and exactly one of x3,x4)
    s1 = torch.sign(Nx-x1)
    s2 = torch.sign(Nx-x2)
    s12 = (s1*s2 -1) / -2.0   
    s3 = torch.sign(Nx-x3)
    s4 = torch.sign(Nx-x4)
    s34 = (s3*s4 -1) / -2.0    
    s_total = s12*s34 # 1 if line segments intersect, 0 otherwise
    keep = torch.nonzero(s_total)
    plot_poly([],im = im,show = False)
    
    keep = keep.detach()
    Nx = Nx[keep[:,0],keep[:,1]]
    Ny = Ny[keep[:,0],keep[:,1]]
    intersections = torch.stack([Nx,Ny],dim = 1)
    plot_poly(intersections,color = (0,255,0),im = im, lines = False,show = False)
    
    # union intersection points to poly1 and poly2 points
    union = torch.cat((poly1,poly2,intersections),dim = 0)
    
    #  maintain an adjacency matrix
    n_elem = union.shape[0]
    p1 = poly1.shape[0]
    p2 = poly2.shape[0]
    
    adj1 = torch.zeros([p1,p1])
    for i in range(-1,p1-1):
        adj1[i,i+1] = 1
        adj1[i+1,i] = 1
    adj2 =  torch.zeros([p2,p2])
    for i in range(-1,p2-1):
        adj2[i,i+1] = 1
        adj2[i+1,i] = 1

    adj = torch.zeros([n_elem,n_elem])
    adj[0:p1,0:p1] = adj1
    adj[p1:p2+p1,p1:p1+p2] = adj2
    
    #plot_adjacency(union,adj,color = (0,0,0),im = None)

    
    # for each intersection, remove 2 connections and add 4
    for i in range(keep.shape[0]):
        xi1 = keep[i,0]
        xi2 = (xi1 - 1)%p1
        xi3 = keep[i,1]
        xi4 =(xi3 - 1)%p2
        
        xi3 = xi3.clone() + p1
        xi4 = xi4.clone() + p1
        
        adj[xi1,xi2] = 0
        adj[xi2,xi1] = 0
        adj[xi3,xi4] = 0
        adj[xi4,xi3] = 0
        
        new_idx = i + p1 + p2
        adj[new_idx,xi1] = 1
        adj[xi1,new_idx] = 1
        adj[new_idx,xi2] = 1
        adj[xi2,new_idx] = 1
        adj[new_idx,xi3] = 1
        adj[xi3,new_idx] = 1
        adj[new_idx,xi4] = 1
        adj[xi4,new_idx] = 1
    

        # deal with pairs of intersections on same line segment
        for j in range(keep.shape[0]):
            if i!=j and (keep[j,0] == keep[i,0] or keep[i,1] == keep[j,1]):

                # connect the intersections to one another
                adj[new_idx,p1+p2+j] = 1
                adj[p1+p2+j,new_idx] = 1
                
                
                # verify that for the two endpoints of the shared segment, only one intersection is connected to each
                if keep[j,0] == keep[i,0]:
                    # if the x coordinate of intersection i is closer to xi1 than intersection j, adjust connections
                    if torch.abs(union[p1+p2+i,0] - xi1) < torch.abs(union[p1+p2+j,0] - xi1): # i is closer
                        con = 1
                    else:
                        con = 0
                    adj[xi1,p1+p2+i] = con
                    adj[p1+p2+i,xi1] = con
                    adj[xi1,p1+p2+j] = 1-con
                    adj[p1+p2+j,xi1] = 1-con
                    adj[xi2,p1+p2+i] = 1-con
                    adj[p1+p2+i,xi2] = 1-con
                    adj[xi2,p1+p2+j] = con
                    adj[p1+p2+j,xi2] = con
                
                elif keep[j,1] == keep[i,1]:
                    # if the x coordinate of intersection i is closer to xi1 than intersection j, adjust connections
                    if torch.abs(union[p1+p2+i,0] - xi3) < torch.abs(union[p1+p2+j,0] - xi3): # i is closer
                        con = 1
                    else:
                        con = 0
                    adj[xi3,p1+p2+i] = con
                    adj[p1+p2+i,xi3] = con
                    adj[xi3,p1+p2+j] = 1-con
                    adj[p1+p2+j,xi3] = 1-con
                    adj[xi4,p1+p2+i] = 1-con
                    adj[p1+p2+i,xi4] = 1-con
                    adj[xi4,p1+p2+j] = con
                    adj[p1+p2+j,xi4] = con
                        
    
    #plot_adjacency(union[p1+p2:,:],adj[p1+p2:,p1+p2:],color = (0,0,0),im = im)

    # find the convex hull of the union of the polygon, and remove these points from the set of overall points
    hull_indices = get_hull(union,indices = True)
    subset_idxs = []
    for i in range(union.shape[0]):
        if i not in hull_indices:
            subset_idxs.append(i)
    subset = union[subset_idxs,:]
    adj_subset = adj[subset_idxs,:][:,subset_idxs]
    
    # plot the intersection
    #im = plot_poly(subset,color = (0,0,0),im = im, lines = False,show = False)
    #plot_adjacency(subset,adj_subset,color = (0,0,0),im = im)

    # repeatedly go through list and remove any points with only one collection
    changed = True
    while changed:
        changed = False
        keep_rows = torch.where(torch.sum(adj_subset,dim = 0) > 1)[0]
        
        if len(keep_rows) < subset.shape[0]:
            changed = True
            
        subset = subset[keep_rows,:]
        adj_subset = adj_subset[keep_rows,:][:,keep_rows]
        

    # order the points in a clockwise ordering 
    polyi = clockify(subset)
    
    # find the area of each of the convex 3 polygons - needs to be differentiable 
    a1 = poly_area(poly1)
    a2 = poly_area(poly2)
    ai = poly_area(polyi)
    
    #print("Poly 1 area: {}".format(a1))
    #print("Poly 2 area: {}".format(a2))
    #print("Intersection area: {}".format(ai))
    iou = ai / (a1 + a2 - ai + 1e-10)
    #print("Polygon IOU: {}".format(iou))
    plot_poly(polyi,color = (0.2,0.7,0.1),im = im, lines = True,text = "Polygon IOU: {}".format(iou))
    
    return iou
    # return
    
def clockify(polygon, clockwise = True):
    """
    polygon - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    clockwise - if True, clockwise, otherwise counterclockwise
    returns - [n_vertices,2] tensor of sorted coordinates 
    """
    
    # get center
    center = torch.mean(polygon, dim = 0)
    
    # get angle to each point from center
    diff = polygon - center.unsqueeze(0).expand([polygon.shape[0],2])
    tan = torch.atan(diff[:,1]/diff[:,0])
    direction = (torch.sign(diff[:,0]) - 1)/2.0 * -np.pi
    
    angle = tan + direction
    sorted_idxs = torch.argsort(angle)
    
    
    if not clockwise:
        sorted_idxs.reverse()
    
    polygon = polygon[sorted_idxs.detach(),:]
    return polygon


poly1 = get_poly(starting_points = 4) * torch.rand(1) + torch.rand(1)
poly2 = get_poly(starting_points = 4)
poly = torch.autograd.Variable(poly1,requires_grad = True)
opt = torch.optim.SGD([poly], lr = 0.05)

for k in range(1000):
    
    
    #riou = raster_poly_IOU(poly1,poly2,scale = 1000)
    #print("Raster IOU: {}".format(riou))

    piou = do_something(poly,poly2)
    opt.zero_grad()
    lp = (1.0 - piou)
    l2 = torch.pow((poly - poly2),2).mean()
    loss = l2 + lp
    loss.backward()
    
    
    opt.step()
    del loss
    
