import sys, os


# An installation agnostic method to find and link to root of the package which is mlfactory
#==========================================================
import re

try: #testing the functions locally without pip install
    import __init__
    cimportpath = os.path.abspath(__init__.__file__)
    if 'extensions' in cimportpath:
        print("Non local usage")
        import mlfactory
        cimportpath = os.path.abspath(mlfactory.__file__)+'/applications/visual_odometry/__init__.py'

except: #testing while mlfactory is installed using pip
    print("Non local usage")
    import mlfactory
    cimportpath = os.path.abspath(mlfactory.__file__)+'/applications/visual_odometry/__init__.py'

idxlist = [m.start() for m in re.finditer(r"/", cimportpath)]
invoking_submodule = cimportpath[idxlist[-2]+1:idxlist[-1]]
print("In visual_odometry/sfm.py got invoking submodule using re",invoking_submodule)
main_package_loc = cimportpath[:cimportpath.rfind('mlfactory')+len('mlfactory')]
print("In visual_odometry/sfm.py got main package location ",main_package_loc)


os.environ['sfm'] = main_package_loc+'/applications/visual_odometry'
os.environ['top'] = main_package_loc
sys.path.append(os.path.join(os.environ['sfm']))
sys.path.append(os.path.join(os.environ['top']))
#==========================================================




from applications.visual_odometry import sfm
import open3d as o3d
import math
#alternative to all the lines above if you dont have mlfactory source code
# pip install -U mlfactory
# from mlfactory.applications.visual_odometry import sfm

import cv2
import numpy as np
from datetime import datetime as dt
import copy






#use superglue False and "cv" is the fastest but often innacurate
#Current calibration settings work well with iphone11 back camera shooting video in portrait mode
#Current algorithm also works better if youre walking fast at a brisk pace instead of panning camera very slowly
#in applications/visual_odometry/sfm.py you can uncomment line 321 in find_pose_change function to get another type of result (dont still understand whats the use of that constraint though)


# pass demo video files like this

#demo_video_file = '/datasets/sample_videos/living_room.MOV'
#rgb_frame_upside_down = True

demo_video_file = '/datasets/sample_videos/cambridge_office.MOV'
rgb_frame_upside_down = False


#Image post processing for changes in visualization flags
iterative_viz = True
inverse_rotate = True



# -----> Algorithm usage variables <----- #
use_superglue = False #uses sift which produce lesser matches but is much faster
triangulate_points = False
essential_estimation = "cv" #"cv", "zhang", "svd"






# -----> Camera variables <----- #
image_width = 320
image_height = 240
fx, fy, cx, cy = 520, 520, image_width//2, image_height//2
K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0,1]])
K_inv = np.linalg.inv(K)




#specify the number of frames in the video you want to process
max_frame = 1800 #500


# ----> Initialising trajectory Variables <----- #
Translation = np.zeros((3, 1))
Rotation = np.eye(3)
count = 0
#this list will store the entire sequence of estimated poses
pose_sequence = [ np.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0] ] ) ]
#initialize the visual odometry class
func = sfm.VisualOdometry()














#initialize the renderer
render = o3d.visualization.rendering.OffscreenRenderer(image_width, image_height)
render.scene.set_background([0.1, 0.2, 0.3, 1.0])  # RGBA


#gets the homogenous transformation matrix describing the pose of the camera with absolute coordinate (0,0,0)
#returns accordingly camera center, eye and look up vectors that define a view exactly 10 units behind the current camera pose
def get_lookats(ext):

    er = ext[0:3,0:3]
    et = np.array([ ext[0,3],ext[1,3],ext[2,3] ])


    C = et #center
    offset = np.array([0,0,-10,1]).dot(np.linalg.inv(ext)) #used to calculate eye vector

    U = [0,1,0]
    U = np.array(U).dot(ext[0:3,0:3].T) #up vector
    
    #print("er ",er)
    return C, U, offset




def process_resize(frame, image_width, image_height): #resize according to the size in which camera was calibrated
    frame = cv2.resize(frame, (image_width, image_height))
    if iterative_viz:
        if rgb_frame_upside_down:
            cv2.imshow("raw rgb ",np.flipud(frame))
        else:
            cv2.imshow("raw rgb ",frame)

        cv2.waitKey(1)
    return frame


def render_passive(img_width, img_height, mesh, mesh_pose_absolute, name):

    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    

    # Optionally set the camera field of view (to zoom in a bit)
    vertical_field_of_view = 45.0  # between 5 and 90 degrees
    aspect_ratio = img_width / img_height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 100.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    render.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)



    intrinsic = o3d.camera.PinholeCameraIntrinsic(image_width, image_height, 520, 520, image_width//2, image_height//2)
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    extrinsic = mesh_pose_absolute


    
    C, U, offset= get_lookats(extrinsic)

    center = C  # look_at target
    eye = [C[0]+offset[0], C[1]+offset[1], C[2]+offset[2]]  # camera position
    up = U  # camera orientation
    render.scene.camera.look_at(center, eye, up)

    
    
    # Add the arrow mesh to the scene.
    # (These are thicker than the main axis arrows, but the same length.)
    for m in mesh:
        render.scene.add_geometry(name, m, mtl)


    # Read the image into a variable
    img_o3d = render.render_to_image()

    # Display the image in a separate window
    # (Note: OpenCV expects the color in BGR format, so swop red and blue.)
    img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGR)
    cv2.imshow("Preview window", img_cv2)
    cv2.waitKey(1)
    return img_cv2


if __name__ == '__main__':


    #pass the video that we just downloaded
    cap = cv2.VideoCapture(demo_video_file)


    ret, key_frame_current = cap.read()
    key_frame_current = process_resize(key_frame_current, image_width, image_height)
    current_frame = key_frame_current.copy()



    while cap.isOpened():
        ret, key_frame_next = cap.read()
        key_frame_next = process_resize(key_frame_next, image_width,image_height)
        #cv2.imshow("frame ",key_frame_next)
        #cv2.waitKey(1)

        next_frame = key_frame_next.copy()


        if ret:

            point_correspondence_cf, point_correspondence_nf = func.extract_correspondences(current_frame,next_frame,use_superglue)
            R, t = func.find_pose_change(K,point_correspondence_cf,point_correspondence_nf,essential_estimation)
            Translation, Rotation, pose_sequence = func.accumulate_poses(Translation, Rotation, t, R, pose_sequence)

            if count==0:
                first_frame = current_frame.copy()
            
            #still work going on to get a corresponding sparse map as well, uncomment, but outputs jibberish for now
            #p3d = func.accumulate_sparse_map(R, t, K, point_correspondence_cf, point_correspondence_nf, pose_sequence)

            

            #plain coordinate frame visualization
            
            coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            coord_mesh.scale(1, center=coord_mesh.get_center()) 
            coord_mesh.transform(pose_sequence[-1])
            

            #tetrahedron mesh visualization
            '''
            coord_mesh = o3d.geometry.TriangleMesh.create_tetrahedron(radius=1.0)
            coord_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(coord_mesh)
            coord_mesh.scale(1, center=coord_mesh.get_center()) 
            coord_mesh.transform(pose_sequence[-1])
            '''
            
            




            if iterative_viz:
                
                #if count%3==0: #make it less cluttered
                prof_camera_traj = render_passive(640, 480, [coord_mesh], pose_sequence[-1], str(count))
                
                dst = cv2.addWeighted(prof_camera_traj, 0.5, key_frame_next, 0.5, 0)
                cv2.imshow("merged viz ",dst)
                cv2.waitKey(1)


            count += 1
            print('# -----> Frame No:'+str(count),'<----- #', dt.now())
            if count==max_frame:
                break
        current_frame = next_frame
    
    #still work going on to get a corresponding sparse map as well, set to true, but outputs jibberish for now
    func.visualize_trajectory(pose_sequence, show_sparse_points=False)

