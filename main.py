import cv2
import ActiveContour
from Utilities.Read_Show import *



imagePath = "./images/example2.jpg"


# Loads the desired image
image = Read_Img( imagePath)

# Creates the snake
snake = ActiveContour.Active_Contour( image)




while( True ):

    # Gets an image of the current state of the snake
    snakeImg = snake.display()
    Show_Img( "Active Contour", snakeImg )
    print(snake.get_length())    #from 1252.5 to 754.08
    # Processes a snake step
    snake_changed = snake.step()

    # Stops looping when ESC pressed
    k = cv2.waitKey(33)
    if k == 27:
        break


cv2.destroyAllWindows()
