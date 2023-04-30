import streamlit as st
import cv2
import numpy as np
# Set page title
st.set_page_config(page_title="DARTS SCORE", page_icon=":pencil2:")

# Define function to apply a filter to the image
def apply_filter(image):
    
    darts=cv2.resize(image,(1400,1400))
    da=darts
   
    img =darts
    # Convert the image to HS7 color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   
    # Define range of green color
    green_lower = (60, 100, 100)
    green_upper = (90, 200, 200)

    # Create a binary mask that marks the pixels corresponding to the object
    mask = cv2.inRange(hsv, green_lower, green_upper)

    ## Apply the mask to the original image to get only white object
    result = cv2.bitwise_and(img, img, mask=mask)
    # Create a mask that selects only the green pixels in the image
    # mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply the mask to the original image using bitwise_and
    masked_img = cv2.bitwise_and(img, img, mask=mask)


    # Find the bounding box of the non-zero pixels in the mask
    y,x = np.nonzero(mask)
    bbox = (x.min(), y.min(), x.max()-x.min(), y.max()-y.min())
    # 
    # Crop the mask from the original image using the bounding box
    cropped_mask = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    cropped_img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    center_x, center_y = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
    # Display the results

  
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Cropped mask', cropped_mask)
   
    cropped_img=cv2.resize(cropped_img,(800,800))
    # print(cropped_img.shape)
    # plt.imshow(cropped_img)

    image=cropped_img
    # image=cv2.resize(image,(800,800))
    blur1 = cv2.GaussianBlur(image, (5,5), 0) 

    # blur1 = cv2.GaussianBlur(img, (3, 3),0)
    # median = cv2.medianBlur(gray, 3)
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(blur1, cv2.COLOR_BGR2HSV)

    # Define range of green color
    green_lower = (25, 40, 0)
    green_upper = (100, 255, 255)

    # Create a binary mask that marks the pixels corresponding to the object
    mask_green = cv2.inRange(hsv_image, green_lower, green_upper)


    # Apply the mask to the original image to extract the object
    # object = cv2.bitwise_and(image, image, mask=mask)

    # Define range of red color
    bag_lower = (15, 30, 180)
    bag_upper = (25,70,255 )
    bag_lower2 = (0, 50, 200)
    bag_upper2 = (230, 90, 255) 
    # Create a binary mask that marks the pixels corresponding to the object
    mask2= cv2.inRange(hsv_image, bag_lower, bag_upper)
    mask3 = cv2.inRange(hsv_image, bag_lower2, bag_upper2)
    mask_bag = cv2.bitwise_or(mask2, mask3)

    # Define range of red color
    red_lower = (0, 50, 50)
    red_upper = (10, 255, 255)
    red_lower2 = (170, 50, 50)
    red_upper2 = (180, 255, 255)

    # Create a binary mask that marks the pixels corresponding to the object
    mask1r = cv2.inRange(hsv_image, red_lower, red_upper)
    mask2r = cv2.inRange(hsv_image, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask1r, mask2r)

    # Define range of black color
    black_lower = (0, 0, 0)
    black_upper = (170, 90,130)

    # Create a binary mask that marks the pixels corresponding to the object
    mask_black = cv2.inRange(hsv_image, black_lower, black_upper)


    mask=  mask_bag + mask_green+ mask_red + mask_black
    # Apply the mask to the original image to extract the object
    mas = cv2.bitwise_and(image, image, mask=mask)
    image=image-mas

    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 155])
    upper_white = np.array([170, 35, 255])

    # Threshold the image to get only white colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Apply the mask to the original image to get only white object
    white = cv2.bitwise_and(image, image, mask=mask_white)
  
    # cv2.imwrite('mask.png',mas)
    gray=cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
    # m=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # plt.imshow(hsv,'gray')
    blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
    median = cv2.medianBlur(gray, 3)

    _, thresh = cv2.threshold(median, 150, 200, cv2.THRESH_BINARY)
    # cv2.imshow("thresh",thresh)
    # Create a structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # # Perform erosion
    # eroded = cv2.erode(thresh, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Perform opening (erosion followed by dilation)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations=1)

    # Perform closing (dilation followed by erosion)
    # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)



    # Perform dilation
    # dilated = cv2.dilate(opening, kernel)
    # closing = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    canny2 = cv2.Canny(opening, 200, 250)
    # result_img=cv2.resize(org_image,(800,800))
    img=cropped_img
    scoress=[]
    # Get the image height and width
    height, width = img.shape[:2]
    height= height + 20
    width = width -3
    z=15
    y=0
    # Get the center of the image
    center = (width // 2, height // 2)
    # center[0] =center[0] -2
    # center[1]=center[1]-2
    # Draw the x-y axis
    cv2.line(img, (center[0], 0), (center[0], height), (0, 255, 0), 2)
    cv2.line(img, (0, center[1]), (width, center[1]), (0, 255, 0), 2)




    # Find the contours of the shape
    contours, _ = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Get the center point of the image
    # height =height-20
    # width =width- 2
    center_point = np.array([int(width/2), int(height/2)])

    # Loop through all the contours and find the nearest point
    for contour in contours:
        # Find the nearest point to the center point of the image
        distances = []
        for point in contour:
            dist = cv2.norm(np.subtract(center_point, point[0]))
            distances.append(dist)
        nearest_point_index = np.argmin(distances)
        nearest_point = tuple(contour[nearest_point_index][0])

        # Draw a line from the center point to the nearest point in the contour
        # cv2.line(img, tuple(center_point), nearest_point, (0, 0, 255), 2)


    # Threshold the image
    thresh = cv2.threshold(opening, 50, 100, cv2.THRESH_BINARY)[1]

    # Find contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    # Draw the contours on the image
    cv2.drawContours(img, contours, -1, (0,0,255), thickness=2)


    # Iterate through all the contours
    for contour in contours:
        # Find the length of the contour
        length = cv2.arcLength(contour, True)
        # print(length)
        # Find the endpoints of the line
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((img.shape[1]-x)*vy/vx)+y)

        # Scale the length of the line
        line_length = int(length * 0.8)  # scaling factor of 0.8

        # Find the unit vector in the direction of the line
        unit_vector = np.array([vx, vy]) / np.sqrt(vx**2 + vy**2)

        # Find the endpoints of the scaled line
        start_point = np.array([x, y]) - unit_vector * line_length /4
        end_point = np.array([x, y]) + unit_vector * line_length /4

        # Draw the line on the image
        cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (0, 255,0 ), 2)


        x=int(end_point[0])

        y=int(end_point[1])
 
        # Calculate the difference between the point and the center
        dx = x - center[0]
        dy = center[1] - y

        # Calculate the angle
        if dx == 0:
            if dy > 0:
                angle = 90
            else:
                angle = 270
        else:
            angle = np.arctan(dy/dx) * 180 / np.pi
            if dx < 0: 
                angle += 180 
            elif dy <0 :
            # else:
                angle += 360
      
        # Calculate the length of the line
        line_length = ((x - center[0])**2 + (y - center[1])**2)**0.5
        linee=round(line_length)

        if angle >= 81 and angle <= 99:
            score = 20
        elif angle >= 316 and angle <= 332:
            score = 15

        elif angle >= 63 and angle <= 80:
            score = 1
        elif angle >= 43 and angle <=62:
            score = 18
        elif angle >= 26.5 and angle <=44:
            score = 4
        elif angle >=9.6 and angle <= 26:
            score = 13
        elif angle <=9.5  or angle >= 351:
       
            score = 6
        elif angle >= 332.5 and angle <= 352:
            score = 10

        elif angle >= 297 and angle <=314:
            score = 2
        elif angle >= 278 and angle <= 298:
            score = 17
        elif angle >= 261 and angle <= 277.5:
            score = 3
        elif angle >= 244 and angle <= 260:
            score = 19
        elif angle >= 225 and angle <= 243:
            score = 7
        elif angle >= 207 and angle <=224:
            score = 16
        elif angle >= 193 and angle <= 206:
            score = 8
        elif angle >= 171 and angle <= 192:
            score = 11
        elif angle >= 153 and angle <= 170:
            score = 14
        elif angle >= 135 and angle <= 152:
            score = 9
        elif angle >= 118 and angle <= 134:
            score = 12
        elif angle >= 99 and angle <=117.6:
            score = 5

        else:
            "offset"


        # Calculate the length of the line
        # line_length = ((x - center[0])**2 + (y - center[1])**2)**0.5
    #     linee=round(line_length)
        if linee >232 and linee <262:#<272:

            score =score *3
        elif linee>390 and linee <400:
            score=score*2
        elif linee >20 and linee <54.2:
            score =25
        elif linee <20 :
            score =50

        if (linee) < 418:

            cv2.line(img, center, (x,y), (255, 255, 0), 2)
            # Draw the point
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
            # cv2.circle(img, (int(start_point[0]), int(start_point[1])), 2, (100, 255, 0), -1)

            # Put the angle on the image
            cv2.putText(da, f"score: {score:} ", ( 15, 15+z),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 250), 4)
            scoress.append(score)
            z+=30
            # Calculate the length of the line
            # line_length = ((x - center[0])**2 + (y - center[1])**2)**0.5

#             print("Line length:", round(line_length))

#             print(angle)

#             # print("Angle:", angle, "degrees")
#             print("Score:", score)

    
    return da


# Define Streamlit app
def app():
    # Set app title
    st.title("Darts Score")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # If image is uploaded
    if uploaded_file is not None:
        # Read image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)

        # Display original image
        st.image(original_image, channels="BGR", caption="Original Image", use_column_width=True)

        # Choose a filter to apply to the image
        # filter_name = st.selectbox("Choose a filter", ["None", "Blur", "Grayscale", "Canny"])

        # Apply filter and display edited image
        edited_image = apply_filter(original_image)
        st.image(edited_image, channels="BGR", caption="Edited Image", use_column_width=True)

# Run Streamlit app
if __name__ == "__main__":
    app()
