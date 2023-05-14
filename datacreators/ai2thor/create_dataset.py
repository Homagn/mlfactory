import math
import numpy as np
import open3d as o3d
from ai2thor.controller import Controller #pip install ai2thor
import copy

import sys
import os
import time

import register_pcds

os.environ['top'] = '../../'
sys.path.append(os.path.join(os.environ['top']))
from visualizers import pointcloud

def to_rad(th):
    return th*math.pi / 180

def get_pose(controller):
    agent_position = controller.last_event.metadata["agent"]["position"]
    agent_rotation = controller.last_event.metadata["agent"]["rotation"]
    
    print("agent current position and rotation ",agent_position, agent_rotation)

def move_precise(controller,dx,dy,dz, sideways_tilt, updown_tilt):
    #camera roll cannot be changed using controller (but pitch and yaw can using controller)
    #but maybe then postprocess and use homography to rotate slight roll angles
    agent_position = controller.last_event.metadata["agent"]["position"]
    agent_rotation = controller.last_event.metadata["agent"]["rotation"]
    
    
    event = controller.step( dict(action='TeleportFull', 
                            position=dict(x=dx,y= dy, z= dz),
                            rotation=dict(x=agent_rotation["x"],y= agent_rotation["y"]+sideways_tilt, z= agent_rotation["z"])),
                            horizon=updown_tilt)
    


    '''
    event = controller.step( dict(action='TeleportFull', 
                            position=dict(x=dx,y= dy, z= dz),
                            rotation=dict(x=agent_rotation["x"],y= agent_rotation["y"], z= agent_rotation["z"])),
                            horizon=updown_tilt)

    event = controller.step( dict(action='RotateLeft', moveMagnitude=sideways_tilt, snapToGrid=False))

    event = controller.step( dict(action='LookUp', moveMagnitude=updown_tilt, snapToGrid=False))
    '''

    

    '''
    event = controller.step( dict(action='FlyTo', 
                            x=agent_position["x"]+dx,y= agent_position["y"]+dy, z= agent_position["z"]+dz,
                            rotation=dict(x=agent_rotation["x"],y= agent_rotation["y"]+sideways_tilt, z= agent_rotation["z"])),
                            horizon=updown_tilt)
    '''

    #event = controller.step( dict(action='FlyUp', moveMagnitude=dy))
    
    print("last action success ",event.metadata["lastActionSuccess"])
    reachables = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    xs = np.unique([float(i['x']) for i in reachables])
    ys = np.unique([float(i['y']) for i in reachables])
    zs = np.unique([float(i['z']) for i in reachables])
    #print("reachable positions ", xs,ys,zs)

    event = controller.step(action="Done")
    time.sleep(0.5)

    return controller


def example(num ,sideways,upwards,forward,yaws,pitches):
    width, height = 600, 600
    fov = 90
    
    
    controller = Controller(agentMode="drone",
                            scene="FloorPlan1",
                            width=width,
                            height=height,
                            fieldOfView=fov,
                            renderDepthImage=True,
                            gridSize=0.05,
                            snapToGrid=False,
                            cameraFarPlane= 25.0)
    
    '''
    #material color and lighting randomization
    controller.step(
                    action="RandomizeMaterials",
                    useTrainMaterials=None,
                    useValMaterials=None,
                    useTestMaterials=None,
                    inRoomTypes=None
                )

    controller.step(
                    action="RandomizeLighting",
                    brightness=(0.5, 1.5),
                    randomizeColor=True,
                    hue=(0, 1),
                    saturation=(0.5, 1),
                    synchronized=False
                )
    '''


    '''
    #ground robot default controller
    controller = Controller(scene="FloorPlan1",
                            width=width,
                            height=height,
                            fieldOfView=fov,
                            renderDepthImage=True)
    '''
    
    #https://github.com/KuoHaoZeng/Visual_Reaction/blob/33614b7b22c2153dc0c847c5b1991540a6b53a36/data/data_mr.py#L108



    '''
    #test case 1 (just translational change with pitch set at 0)

    #1
    if num==1:
        controller = move_precise(controller,0.62,1.7,0.33, 0,0) #global - sideways,upwards,forwards | sideways tilt, updown tilt
    #2
    if num==2:
        controller = move_precise(controller,0.52,1.6,0.43, 0,0) #global - sideways,upwards,forwards
    '''


    '''
    #test case 2 (just translational change with pitch set at non zero value)

    #1
    if num==1:
        controller = move_precise(controller,0.72,1.8,0.33, -10,0) #global - sideways,upwards,forwards
    #2
    if num==2:
        controller = move_precise(controller,0.62,1.7,0.43, -10,0) #global - sideways,upwards,forwards
    '''

    '''
    #test case 3 (translational change with yaw difference between non zero values)

    #1
    if num==1:
        controller = move_precise(controller,0.72,1.8,0.33, -10,0) #global - sideways,upwards,forwards
    #2
    if num==2:
        controller = move_precise(controller,0.62,1.7,0.43, -20,0) #global - sideways,upwards,forwards
    '''

    '''
    #test case 4 (translational change with yaw and pitch difference between non zero values)

    #1
    if num==1:
        controller = move_precise(controller,0.72,1.8,0.33, -10, 10) #global - sideways,upwards,forwards
    #2
    if num==2:
        controller = move_precise(controller,0.62,1.7,0.43, -20, 20) #global - sideways,upwards,forwards
    '''
    
    if num==1:
        controller = move_precise(controller,sideways[0],upwards[0],forward[0], yaws[0], pitches[0]) #global - sideways,upwards,forwards
    if num==2:
        controller = move_precise(controller,sideways[1],upwards[1],forward[1], yaws[1], pitches[1]) #global - sideways,upwards,forwards
    

    
    
    event = controller.step(action="Done")





    
    #getting camera params

    # Convert fov to focal length
    focal_length = 0.5 * width * math.tan(to_rad(fov/2))
    print("derived focal length ",focal_length)
    fx, fy, cx, cy = (focal_length, focal_length, width/2, height/2)

    camera_params = {}
    camera_params["fx"], camera_params["fy"], camera_params["centerX"], camera_params["centerY"] = fx, fy, cx, cy 
    #assuming scalex=scaley and width=height
    camera_params["scalingFactor"]= width



    #getting state data

    rgb = copy.copy(event.frame.astype(np.uint8))
    dep = copy.copy(event.depth_frame)



    #visualize pointcloud using open3d
    pointcloud.show_pcd_from_rgbd_native(rgb,dep,camera_params,save_loc = str(num)+".pcd")
    print("max depth of the pointcloud ",np.max(dep))
    
    #visualize using my function
    #pointcloud.show_pcd_from_rgbd(rgb,255.0*dep/np.max(dep),camera_params,save_loc = "1.pcd")
    



if __name__ == "__main__":
    
    
    #test case 1 (works perfectly)/ things get bad when starting yaw and pitch angles are non 0
    sideways = [0.62,0.52]
    upwards = [1.7,1.6]
    forward = [0.33,0.43]

    yaws = [0,0]
    pitches = [0,0]
    

    '''
    #test case 2 
    sideways = [0.72,0.62]
    upwards = [1.8,1.7]
    forward = [0.33,0.43]

    yaws = [-10,-20]
    pitches = [10,20]
    '''


    '''
    #test case 3
    sideways = [0.62,0.42]
    upwards = [1.5,1.4]
    forward = [0.23,0.33]

    yaws = [5, 20]
    pitches = [-2,10]
    '''


    example(1,sideways,upwards,forward,yaws,pitches)
    example(2,sideways,upwards,forward,yaws,pitches)

    print("Now testing registration with known information ...")
    
    register_pcds.register("1.pcd","2.pcd", 
        pitch_change = pitches[1]-pitches[0],
        yaw_change = yaws[1]-yaws[0],
        sideways_change = sideways[1]-sideways[0],
        upwards_change = upwards[1]-upwards[0],
        forward_change = forward[1]-forward[0])
