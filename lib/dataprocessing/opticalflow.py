import cv2
import numpy as np

class OpticalFlow:
    """Class representing a module to compute optical flow images

    This class uses opencv to compute the optical flow image between to consecutive
    frames.

    Methods:
        optical_flow(prev, curr)
            compute the optical flow between two images
    """


    def optical_flow(self, prev: np, curr: np):
        """computes the optical flow between two consecutive video frames

        It uses opencv to compute the difference between two images ecoded as the
        optical flow and represents this using an RBG images using the intensities
        and color.

        Paramters:
            prev (np): previous frame as grayscale image
            curr (np): current frame as grayscale image

        Returns:
            np: optical flow rgb image of the same height and width as the input
                images
        """
        
        # hsv matrix to store the optical flow values
        hsv = np.zeros([*prev.shape,3], dtype=np.uint8)
        hsv[...,1] = 255 # saturation

        # compute the optical flow between the two
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # process the optical flow
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        # interprete the angle and orientations from the optical flow as colors
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        of_image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        return of_image
