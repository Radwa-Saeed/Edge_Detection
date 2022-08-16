from Utilities.filters import *

def hough_line(img, min_line_votes=200, cannyThresh1 = 50, cannyThresh2 =150):
        import cv2
        import numpy as np

        #applying canny edge detector with thresholds for hysteresis 50-150
        edges_image = cv2.Canny(img, cannyThresh1, cannyThresh2)    
        
        #getting the dimensions of the resulted image 
        height =  edges_image.shape[0]
        width = edges_image.shape[1]

        #calculating the diagonal length --> this is the maximum value that rho can be
        diagonal_len = int(round(np.hypot(width, height)))

        #calculating the range of thetas and rhos and initializing the accumulator
        thetas = np.deg2rad(np.arange(-90.0, 90.0, 1))
        rs = np.arange(-diagonal_len, diagonal_len, 1 )
        number_of_thetas = len(thetas)

        #initializing the accumulator space with zeros
        accumulator = np.zeros((2 * diagonal_len, number_of_thetas))

        #calculating the cosins and sins of the thetas 
        cos_t = np.cos(thetas)      #cos_t is array of thetas cos
        sin_t = np.sin(thetas)      #sin_t is array of thetas sin

        #y_idxs, x_idxs = np.nonzero(edges_image)

        #getting the indecies of the edges in image 
        edge_points = np.argwhere(edges_image != 0)

        #calculating the rho values for each edge point at each theta
        #diagonal_len is added for a positive index
        rho_values = diagonal_len + np.matmul(edge_points, np.array([sin_t, cos_t]))

        #arranging the theta values  to be repeated form 0 to 180 
        theta_values = np.tile(np.arange(0,number_of_thetas,1) , rho_values.shape[0])

        #making the rho values 1-D array 
        rho_values = rho_values.ravel()

        #incrementing the accumulator by one at each rho value, theta
        for i in range(len(rho_values)):
            accumulator[int(round(rho_values[i])),theta_values[i]] += 1

        # for i in range(len(rho_values)):
        #     for t_idx in range(number_of_thetas):
        #         accumulator[int(round(rho_values[i][t_idx])),t_idx] += 1

        # Vote in the hough accumulator
        # for i in range(len(x_idxs)):
        #     x = x_idxs[i]
        #     y = y_idxs[i]

        #     for t_idx in range(number_of_thetas):
        #         # Calculate rho. diagonal_len is added for a positive index
        #         rho = diagonal_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
        #         accumulator[rho, t_idx] += 1
            
        #return accumulator, thetas, rs

        # the satisfying lines are the indecies in the accumulator which incremented to be more than the minimum line votes we decided
        satisfying_lines = accumulator > min_line_votes

        #creating mesh of thetas and rs 
        x, y = np.meshgrid(thetas,rs)

        #dstack after the mesh so now we have all possible conpinations of rho and theta
        #getting the rho and theta of the satisfying lines only
        lines = np.dstack((x, y))[satisfying_lines]
      
        lines_img = np.zeros_like(edges_image)

        for  theta, rho in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            lines_img = cv2.line(lines_img, (x1, y1), (x2, y2), 255, 2)

        return lines_img



def hough_circles(img, n_circles=2, minRadius=20, maxRadius=30, cannyThresh1 = 50, cannyThresh2 =150):
    import cv2
    import numpy as np

    edges_image = cv2.Canny(img, cannyThresh1, cannyThresh2)
    height =  edges_image.shape[0]
    width = edges_image.shape[1]  

    w = np.arange(0, width, 1)
    h = np.arange(0, height, 1)

    diagonal = int(round(np.hypot(width, height)))
    r = np.arange(0, diagonal, 1)
    

    # Hough accumulator array
    accumulator = np.zeros((len(w), len(h), len(r)), dtype=np.uint16)

    y_idxs, x_idxs = np.nonzero(edges_image)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):

        x = x_idxs[i]
        y = y_idxs[i]

        for x_counter in range(len(w)):
            for y_counter in range(len(h)):
                radius = int(((x - x_counter)**2 + (y - y_counter)**2)**0.5)
                if minRadius <= radius <= maxRadius:
                    try:
                        accumulator[x_counter, y_counter,  radius] += 1
                        #print("NEW VALUE ADDED TO ACCUMULATOR")
                    except:
                        print("PASSED")
                        pass

    highly_voted_circles = np.max(accumulator, axis=2)

    highly_voted_circles = np.sort(highly_voted_circles, axis=None, kind="mergesort")[::-1][:n_circles]

    satisfying_circles = (accumulator >= highly_voted_circles[-1]) & (accumulator <= highly_voted_circles[0])

    circles = np.argwhere(satisfying_circles)

    circles_img = np.zeros_like(edges_image)

    for (x, y, redius) in circles:
        print(x,y,redius)
        circles_img = cv2.circle(circles_img, (x, y),redius, 255 , 2)

    return circles_img



def edgeSuperimpose(image, edges, color = "green"):

    colors = {"red":(255,0,0),
              "green":(0,255,0),
              "blue":(0,0,255)
             }
    
    #check if the color user enterd is in colors Dictionary
    if color.lower() not in colors.keys():
        print("The color you enterd is not valid, Default color used instead")
        color = "green"

    else:
        color = color.lower()

    #copy the image 
    imposed_image = np.copy(image)
    
    #check if the image is rgb
    if image.shape[-1] == 3:
        imposed_image[edges == 255] = colors.get(color)

    #if the image is grey image, it will be superimposed by white color
    elif image.shape[-1] == 2:
        imposed_image[edges == 255] = 255

    else:
        print("Image is not RGB or Grey level image")

    return imposed_image